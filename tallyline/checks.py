"""Checks on one case: facts, document agreement, coverage, warning signs, past cases, risk.

The document-agreement rules and the risk score are pure functions. The language model
is used only where wording must be understood.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from typing import Any, Callable

from . import db, llm, risk_model
from .case_types import CaseType, load_case_type
from .settings import Settings

DOC_FIELDS = {
    "statement": ["incident_date", "location", "incident_description", "loss_detail", "provider", "incident_report_filed"],
    "assessment_report": ["loss_detail", "items_assessed", "estimate"],
    "invoice": ["items_billed", "invoice_total", "invoice_date", "provider"],
    "incident_report": ["incident_date", "location"],
}
LIST_FIELDS = {"items_assessed", "items_billed"}
STOP_WORDS = {"a", "an", "the", "of", "at", "to", "and", "on", "in", "my", "was", "while", "i", "it", "for", "with", "that", "by", "as"}

GENERAL_TERMS = {"provider": "Service provider", "items": "Items", "loss_detail": "Loss detail"}


# ----- value parsing (pure) -----


def parse_date(value: Any) -> date | None:
    """Read a date from model output or document text."""
    if value is None:
        return None
    text = str(value).strip()
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if match:
        try:
            return date.fromisoformat(match.group())
        except ValueError:
            return None
    for fmt in ("%d %B %Y", "%B %d, %Y", "%d/%m/%Y", "%d %b %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_amount(value: Any) -> float | None:
    """Read an amount, ignoring currency words and thousands separators."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"\d[\d,]*(?:\.\d+)?", str(value))
    return float(match.group().replace(",", "")) if match else None


def parse_bool(value: Any) -> bool | None:
    """Read a yes or no value."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower() if value is not None else ""
    if text in {"yes", "true", "filed", "y"}:
        return True
    if text in {"no", "false", "not filed", "n", "none"}:
        return False
    return None


def as_list(value: Any) -> list[str]:
    """Read a list of item names, accepting a comma-separated string."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def word_set(text: str) -> set[str]:
    """Lower-case words of a phrase without common small words."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP_WORDS}


def similar(a: str, b: str, threshold: float = 0.6) -> bool:
    """True if two short phrases share most of their words."""
    words_a, words_b = word_set(a), word_set(b)
    if not words_a or not words_b:
        return False
    return len(words_a & words_b) / len(words_a | words_b) >= threshold


# ----- document agreement rules (pure) -----


def _mismatch(kind: str, description: str, documents: list[str], passages: list[str | None]) -> dict[str, Any]:
    return {"type": kind, "description": description, "documents": documents, "passages": [p for p in passages if p]}


def rule_date_order(
    policy_start: date | None, incident: date | None, invoice: date | None
) -> list[str]:
    """Policy start must not follow the incident, and the incident must not follow the invoice."""
    problems = []
    if policy_start and incident and incident < policy_start:
        problems.append(f"The incident date {incident} is before the policy start date {policy_start}.")
    if incident and invoice and invoice < incident:
        problems.append(f"The invoice date {invoice} is before the incident date {incident}.")
    return problems


def rule_early_incident(policy_start: date | None, incident: date | None, days: int) -> str | None:
    """The incident happened soon after the policy started."""
    if policy_start and incident and timedelta(0) <= incident - policy_start < timedelta(days=days):
        return f"The incident on {incident} is {(incident - policy_start).days} days after the policy start date {policy_start}."
    return None


def rule_loss_detail(statement_value: Any, assessment_value: Any, label: str) -> str | None:
    """The statement and the assessment report must describe the same loss detail."""
    if not statement_value or not assessment_value:
        return None
    if similar(str(statement_value), str(assessment_value)):
        return None
    return (f"The statement and the assessment report describe different details of the loss. "
            f"{label}: the statement says \"{statement_value}\" but the assessment report says \"{assessment_value}\".")


def rule_billed_items(billed: list[str], assessed: list[str]) -> list[str]:
    """Return billed items that do not appear among the assessed items."""
    if not assessed:
        return []
    return [item for item in billed if not any(similar(item, other) for other in assessed)]


def rule_invoice_total(total: float | None, estimate: float | None, ratio: float) -> str | None:
    """The invoice total must not be far above the estimate."""
    if total is None or not estimate or total <= ratio * estimate:
        return None
    return f"The invoice total {total:.2f} is {total / estimate:.2f} times the estimate {estimate:.2f}, above the limit of {ratio}."


def rule_incident_report(says_filed: bool | None, report_exists: bool, label: str) -> str | None:
    """A statement that a report was filed must agree with the case file."""
    if says_filed is True and not report_exists:
        return f"The statement says a {label} was filed, but the case file has none."
    if says_filed is False and report_exists:
        return f"The statement says no {label} was filed, but the case file has one."
    return None


def rule_provider_repeat(
    provider: str | None, incident: date | None, past_cases: list[dict[str, Any]], case_type: str,
    threshold: int, recent_days: int,
) -> list[dict[str, Any]]:
    """Return recent past cases of the same provider, if there are at least `threshold` of them."""
    if not provider or not incident:
        return []
    recent = []
    for record in past_cases:
        when = parse_date(record.get("incident_date"))
        same_type = record.get("case_type") == case_type
        same = str(record.get("provider", "")).strip().lower() == provider.strip().lower()
        if same_type and same and when and timedelta(0) < incident - when <= timedelta(days=recent_days):
            recent.append(record)
    return recent if len(recent) >= threshold else []


def run_rules(
    facts: dict[str, dict[str, Any]], sources: dict[str, dict[str, str]], policy_start: str | None,
    documents_present: set[str], past_cases: list[dict[str, Any]], settings: Settings, case_type: str,
    terms: dict[str, str] | None = None, report_label: str = "incident report",
) -> list[dict[str, Any]]:
    """Apply every document-comparison rule. Each mismatch names its documents and passages."""
    terms = terms or GENERAL_TERMS
    statement = facts.get("statement", {})
    assessment = facts.get("assessment_report", {})
    invoice = facts.get("invoice", {})

    def source(doc: str, field: str) -> str | None:
        return sources.get(doc, {}).get(field)

    start = parse_date(policy_start)
    incident = parse_date(statement.get("incident_date")) or parse_date(facts.get("incident_report", {}).get("incident_date"))
    invoice_date = parse_date(invoice.get("invoice_date"))
    found = []
    for problem in rule_date_order(start, incident, invoice_date):
        found.append(_mismatch("date_order", problem, ["statement", "invoice"],
                               [source("statement", "incident_date"), source("invoice", "invoice_date")]))
    early = rule_early_incident(start, incident, settings.early_incident_days)
    if early:
        found.append(_mismatch("early_incident", early, ["statement"], [source("statement", "incident_date")]))
    loss = rule_loss_detail(statement.get("loss_detail"), assessment.get("loss_detail"), terms["loss_detail"])
    if loss:
        found.append(_mismatch("loss_detail", loss, ["statement", "assessment_report"],
                               [source("statement", "loss_detail"), source("assessment_report", "loss_detail")]))
    extra = rule_billed_items(as_list(invoice.get("items_billed")), as_list(assessment.get("items_assessed")))
    if extra:
        found.append(_mismatch(
            "billed_items", f"The invoice bills {terms['items'].lower()} that do not appear in the assessment report: {', '.join(extra)}.",
            ["invoice", "assessment_report"], [source("invoice", "items_billed"), source("assessment_report", "items_assessed")],
        ))
    total = rule_invoice_total(parse_amount(invoice.get("invoice_total")), parse_amount(assessment.get("estimate")),
                               settings.invoice_over_estimate_ratio)
    if total:
        found.append(_mismatch("invoice_total", total, ["invoice", "assessment_report"],
                               [source("invoice", "invoice_total"), source("assessment_report", "estimate")]))
    report = rule_incident_report(parse_bool(statement.get("incident_report_filed")),
                                  "incident_report" in documents_present, report_label)
    if report:
        found.append(_mismatch("incident_report", report, ["statement"], [source("statement", "incident_report_filed")]))
    provider = invoice.get("provider") or statement.get("provider")
    recent = rule_provider_repeat(provider, incident, past_cases, case_type,
                                  settings.provider_repeat_threshold, settings.provider_recent_days)
    if recent:
        ids = [f"past_cases:{r.get('past_case_id') or r.get('id')}:0" for r in recent[:3]]
        found.append(_mismatch(
            "provider_repeat",
            f"The same {terms['provider'].lower()} appears in an unusually large number of recent cases: {provider}, "
            f"{len(recent)} past cases within {settings.provider_recent_days} days before this incident.",
            ["invoice"], [source("invoice", "provider"), *ids],
        ))
    return found


# ----- risk score (pure) -----


def risk_score(
    probability: float | None, n_mismatches: int, n_flags: int, share_bad_past: float, settings: Settings
) -> dict[str, Any]:
    """Weighted sum of the four parts, 0 to 100. Without a risk model, its weight is shared out."""
    weights = dict(settings.risk_weights)
    if probability is None:
        spare = weights.pop("model")
        total = sum(weights.values())
        weights = {key: value + spare * value / total for key, value in weights.items()}
    values = {
        "model": probability if probability is not None else 0.0,
        "mismatches": min(n_mismatches / settings.risk_mismatches_full, 1.0),
        "red_flags": min(n_flags / settings.risk_red_flags_full, 1.0),
        "past_cases": share_bad_past,
    }
    labels = {
        "model": f"Risk model probability {values['model']:.2f}",
        "mismatches": f"{n_mismatches} document mismatch(es)",
        "red_flags": f"{n_flags} warning sign(s)",
        "past_cases": f"{share_bad_past:.0%} of similar past cases were rejected or investigated",
    }
    parts = [
        {"name": key, "label": labels[key], "weight": round(weight, 3), "points": round(values[key] * weight * 100, 1)}
        for key, weight in weights.items()
    ]
    score = round(sum(values[key] * weight * 100 for key, weight in weights.items()), 1)
    level = "High" if score >= settings.risk_level_high else "Medium" if score >= settings.risk_level_medium else "Low"
    return {"score": score, "level": level, "parts": parts}


# ----- loading a case -----


def load_case(case_id: str, settings: Settings) -> dict[str, Any]:
    """Return the case record, its case type and its document texts."""
    rows = db.query(settings, "SELECT * FROM cases WHERE id = ?", (case_id,))
    if not rows:
        raise KeyError(f"Unknown case {case_id}")
    case = rows[0]
    case_type = load_case_type(case["case_type"], settings)
    folder = settings.cases_dir / case_id
    documents = {
        doc: (folder / f"{doc}.md").read_text(encoding="utf-8")
        for doc in case_type.documents
        if (folder / f"{doc}.md").exists()
    }
    case["features"] = json.loads(case.pop("features_json") or "{}")
    return {"record": case, "case_type": case_type, "documents": documents}


def case_passages(case_id: str, settings: Settings) -> list[dict[str, Any]]:
    """All indexed passages of one case, in document order."""
    rows = db.query(settings, "SELECT id, text, metadata_json FROM passages WHERE index_name = 'case_docs' AND id LIKE ?",
                    (f"case_docs:{case_id}:%",))
    for row in rows:
        row["metadata"] = json.loads(row.pop("metadata_json"))
    rows.sort(key=lambda r: (r["metadata"].get("doc_type", ""), int(r["id"].rsplit(":", 1)[-1])))
    return rows


def _alias_block(passages: list[Any], prefix: str, limit: int = 700) -> tuple[str, dict[str, Any]]:
    """Give passages short labels for a prompt. Returns (text, label -> passage)."""
    lines, aliases = [], {}
    for number, passage in enumerate(passages, start=1):
        label = f"{prefix}{number}"
        aliases[label] = passage
        text = passage["text"] if isinstance(passage, dict) else passage.text
        lines.append(f"[{label}] {text[:limit]}")
    return "\n\n".join(lines), aliases


# ----- checks that use the language model -----


def _field_lines(fields: list[str], case_type: CaseType) -> str:
    terms = case_type.terms
    help_text = {
        "incident_date": "date of the incident, as YYYY-MM-DD",
        "location": "place where the incident happened",
        "incident_description": "what happened, in one sentence",
        "loss_detail": f"the {terms['loss_detail'].lower()}",
        "provider": f"name of the {terms['provider'].lower()}",
        "incident_report_filed": f"true if the document says a {case_type.documents.get('incident_report', 'report').lower()} was filed, false if it says not",
        "items_assessed": f"array of the {terms['items'].lower()} assessed",
        "items_billed": "array of the item names in the invoice table, without the Total row",
        "estimate": "the estimate amount",
        "invoice_total": "the Total amount",
        "invoice_date": "the invoice date, as YYYY-MM-DD",
    }
    return "\n".join(f'- "{field}": {help_text[field]}' for field in fields)


def _find_source(value: Any, doc_passages: list[dict[str, Any]]) -> str | None:
    """The passage that contains a fact, or the document's first passage."""
    if not doc_passages:
        return None
    probe = as_list(value)[0] if isinstance(value, list) and value else value
    probe_text = str(probe).lower() if probe not in (None, True, False) else ""
    for passage in doc_passages:
        if probe_text and probe_text in passage["text"].lower():
            return passage["id"]
    return doc_passages[0]["id"]


def extract_facts(
    case: dict[str, Any], settings: Settings, progress: Callable[[str], None] = lambda _: None
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]], list[str]]:
    """Read each document with the language model. Returns (facts, sources, errors)."""
    case_type: CaseType = case["case_type"]
    passages = case_passages(case["record"]["id"], settings)
    prompt_text = settings.prompt("extract_facts")
    facts: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    for doc, text in case["documents"].items():
        progress(f"Reading {case_type.documents[doc]}")
        fields = [f for f in DOC_FIELDS.get(doc, []) if f in case_type.fact_fields]
        prompt = prompt_text.format(
            document_label=case_type.documents[doc], field_lines=_field_lines(fields, case_type), document=text[:6000]
        )
        try:
            value = llm.complete_json(prompt, settings, max_tokens=300)
        except llm.LLMError as exc:
            errors.append(f"{case_type.documents[doc]}: {exc}")
            continue
        facts[doc] = {field: value.get(field) for field in fields}
        doc_passages = [p for p in passages if p["metadata"].get("doc_type") == doc]
        sources[doc] = {}
        for field in fields:
            found = _find_source(value.get(field), doc_passages)
            if found:
                sources[doc][field] = found
    return facts, sources, errors


def check_coverage(case: dict[str, Any], facts: dict[str, dict[str, Any]], settings: Settings) -> dict[str, Any]:
    """Decide covered, excluded, partly or unclear from this case type's policy passages only."""
    from . import rag

    statement = facts.get("statement", {})
    incident = str(statement.get("incident_description") or case["documents"].get("statement", "")[:600])
    passages = rag.retrieve(
        "knowledge", incident, {"case_type": case["case_type"].name, "source": "policy"},
        settings=settings, min_score=rag.NO_MIN_SCORE,
    )
    if not passages:
        return {"status": "unclear", "reason": "No policy passages were found.", "passages": [], "clauses": []}
    block, aliases = _alias_block(passages, "P")
    items = as_list(facts.get("assessment_report", {}).get("items_assessed")) or as_list(facts.get("invoice", {}).get("items_billed"))
    prompt = settings.prompt("coverage").format(incident=incident, items=", ".join(items) or "not stated", passages=block)
    try:
        value = llm.complete_json(prompt, settings, required_keys=["status"], max_tokens=200)
    except llm.LLMError as exc:
        return {"status": "unclear", "reason": str(exc), "passages": [], "clauses": []}
    status = str(value.get("status", "")).strip().lower()
    if status not in {"covered", "excluded", "partly", "unclear"}:
        status = "unclear"
    cited = [aliases[label] for label in as_list(value.get("passages")) if label in aliases]
    clauses = [p.metadata.get("clause_id") for p in cited if p.metadata.get("clause_id")]
    reason = str(value.get("reason", ""))
    checked = reconcile_coverage(status, clauses, case["case_type"].clause_prefixes)
    if checked != status:
        reason = f"The clauses cited do not support the first answer, \"{status}\", so the status was changed to \"{checked}\"."
    return {"status": checked, "reason": reason, "passages": [p.id for p in cited], "clauses": clauses}


def reconcile_coverage(status: str, clauses: list[str], prefixes: dict[str, str] | None) -> str:
    """Correct a status that the clauses the model cited cannot support.

    The model often cites the right clause but states the wrong status, and a wrong
    "excluded" leads to a recommendation to reject. Needs the case type's `clause_prefixes`;
    without them, or without cited clause IDs, the status is left as it is.
    """
    if not prefixes or not clauses:
        return status

    def cited(kind: str) -> bool:
        prefix = prefixes.get(kind)
        return bool(prefix) and any(clause.startswith(prefix) for clause in clauses)

    if status == "excluded" and not cited("exclusion"):
        if cited("limit"):
            return "partly"
        return "covered" if cited("cover") else "unclear"
    if status == "covered" and cited("limit"):
        return "partly"
    if status == "partly" and not cited("limit"):
        return "covered" if cited("cover") else "unclear"
    return status


def check_red_flags(
    case: dict[str, Any], facts: dict[str, dict[str, Any]], mismatches: list[dict[str, Any]], settings: Settings
) -> list[dict[str, Any]]:
    """Retrieve warning signs for the case and its mismatches; the language model picks those that apply."""
    from . import rag

    case_type: CaseType = case["case_type"]
    filters = {"case_type": [case_type.name, "general"], "source": "red_flags"}
    statement = case["documents"].get("statement", "")
    guide = rag.retrieve_many("knowledge", red_flag_queries(statement, mismatches), filters, settings, limit=8)
    if not guide:
        return []
    guide_block, _ = _alias_block(guide, "G", limit=300)
    case_rows = case_passages(case["record"]["id"], settings)
    statement_rows = [p for p in case_rows if p["metadata"].get("doc_type") == "statement"]
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    def raise_flag(text: str, case_passage: str | None) -> None:
        """Raise the warning sign a text matches best, if it matches well enough."""
        passage, score = rag.best_match(text, guide, settings)
        if score < settings.red_flag_min_score:
            db.log_removal(settings, case["record"]["id"], "warning sign", text,
                           f"closest warning sign scored {score:.2f}, below {settings.red_flag_min_score}")
        elif passage.id not in seen:
            seen.add(passage.id)
            results.append({
                "flag_id": passage.metadata.get("flag_id") or passage.id,
                "title": passage.text.split("\n", 1)[0].lstrip("# ").strip(),
                "guide_passage": passage.id,
                "case_passage": case_passage,
                "evidence": text,
            })

    # A mismatch is already a checked fact, so it needs no model to raise its warning sign.
    for mismatch in mismatches:
        raise_flag(mismatch["description"], next(iter(mismatch["passages"]), None))

    # The model picks a numbered sentence rather than quoting, so the evidence is always the
    # document's own words; asked to quote, it pasted whole paragraphs. It decides which
    # sentences look wrong, but it often names the wrong sign and sometimes picks an ordinary
    # sentence, so the reranker chooses the sign and can turn the sentence down.
    sentences = {f"S{number}": item for number, item in enumerate(case_sentences(statement_rows[:3]), start=1)}
    prompt = settings.prompt("red_flags").format(
        guide=guide_block, sentences="\n".join(f"[{label}] {text}" for label, (text, _) in sentences.items()) or "None",
    )
    try:
        value = llm.complete_json(prompt, settings, required_keys=["flags"], max_tokens=200)
    except llm.LLMError:
        return results
    entries = value.get("flags") if isinstance(value.get("flags"), list) else []
    picked = {str(entry.get("sentence")).strip("[] ") for entry in entries if isinstance(entry, dict)}
    for label, (text, case_passage) in sentences.items():
        if label in picked:
            raise_flag(text, case_passage)
    return results


def _narrative_sentences(text: str) -> list[str]:
    """Sentences of a document's running text, without headings, tables or listed details.

    "Label: value" lines must go too. Text from a PDF has no full stop after them, so they
    would otherwise run on into the sentence that follows and hide it.
    """
    body = re.sub(r"(?m)^(#.*|- .*|\|.*|[A-Z][A-Za-z' ]{0,30}: .*)$", "", text)
    sentences = (" ".join(s.split()) for s in re.split(r"(?<=[.!?])\s+", body))
    return [s for s in sentences if len(s.split()) >= 5]


def case_sentences(passages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Each distinct sentence in these passages, with the ID of the passage it came from."""
    found: dict[str, str] = {}
    for passage in passages:
        for sentence in _narrative_sentences(passage["text"]):
            found.setdefault(sentence, passage["id"])
    return list(found.items())


def red_flag_queries(statement_text: str, mismatches: list[dict[str, Any]]) -> list[str]:
    """One search per mismatch and per sentence of the statement that could show a warning sign."""
    return [m["description"] for m in mismatches] + _narrative_sentences(statement_text)[-4:]


def similar_past_cases(
    case: dict[str, Any], facts: dict[str, dict[str, Any]], settings: Settings, n: int = 5
) -> tuple[list[dict[str, Any]], float]:
    """Top past cases by retrieval, plus exact matches on provider. Returns (list, share rejected or investigated)."""
    from . import rag

    case_type: CaseType = case["case_type"]
    statement = facts.get("statement", {})
    provider = facts.get("invoice", {}).get("provider") or statement.get("provider") or case["record"].get("provider")
    description = str(statement.get("incident_description") or case["documents"].get("statement", "")[:400])
    question = f"{description} {statement.get('location') or ''} {provider or ''}".strip()
    retrieved = rag.retrieve("past_cases", question, {"case_type": case_type.name}, settings=settings,
                             top_n=n, min_score=rag.NO_MIN_SCORE)
    records = {r["id"]: r for r in db.query(settings, "SELECT * FROM past_cases WHERE case_type = ?", (case_type.name,))}
    term = case_type.terms["provider"]
    results, seen = [], set()

    def add(past_id: str, retrieved_here: bool) -> None:
        record = records.get(past_id)
        if not record or past_id in seen:
            return
        seen.add(past_id)
        same = bool(provider) and str(record["provider"]).lower() == str(provider).lower()
        shared = sorted(word_set(description) & word_set(record["summary"]))[:5]
        why = []
        if same:
            why.append(f"Same {term.lower()}: {record['provider']}")
        if shared:
            why.append("Similar wording: " + ", ".join(shared))
        results.append({
            "past_case_id": past_id, "outcome": record["outcome"], "provider": record["provider"],
            "amount": record["amount"], "currency": record["currency"], "summary": record["summary"],
            "passage": f"past_cases:{past_id}:0", "why": "; ".join(why) or "Similar account of the incident",
            "same_provider": same, "retrieved": retrieved_here,
        })

    exact = [r for r in records.values() if provider and str(r["provider"]).lower() == str(provider).lower()]
    retrieved_ids = [p.metadata.get("past_case_id") for p in retrieved]
    newest_exact = sorted(exact, key=lambda r: r["incident_date"] or "", reverse=True)
    for past_id in [i for i in retrieved_ids if i in {r["id"] for r in exact}]:
        add(past_id, True)
    # Search alone can fill the list, so keep room for the newest exact matches.
    for record in newest_exact[:2]:
        if len(results) < min(2, n):
            add(record["id"], False)
    for past_id in retrieved_ids:
        if len(results) < n:
            add(past_id, True)
    for record in newest_exact:
        if len(results) < n:
            add(record["id"], False)
    bad = sum(1 for r in results if r["outcome"] in {"rejected", "sent_for_investigation"})
    return results, (bad / len(results) if results else 0.0)


def run_checks(
    case_id: str, settings: Settings, progress: Callable[[str], None] = lambda _: None
) -> dict[str, Any]:
    """Run every check on one case and return the findings."""
    case = load_case(case_id, settings)
    case_type: CaseType = case["case_type"]
    record = case["record"]
    facts, sources, errors = extract_facts(case, settings, progress)

    progress("Comparing the documents")
    past = db.query(settings, "SELECT id, case_type, provider, incident_date FROM past_cases")
    mismatches = run_rules(
        facts, sources, record.get("policy_start"), set(case["documents"]), past, settings, case_type.name,
        case_type.terms, case_type.documents.get("incident_report", "incident report").lower(),
    )
    progress("Checking the policy wording")
    coverage = check_coverage(case, facts, settings)
    progress("Checking the warning signs")
    flags = check_red_flags(case, facts, mismatches, settings)
    progress("Searching past cases")
    similar_cases, share_bad = similar_past_cases(case, facts, settings)

    predicted = risk_model.predict_probability(case_type, record["features"], settings)
    probability, factors = predicted if predicted else (None, [])
    risk = risk_score(probability, len(mismatches), len(flags), share_bad, settings)
    risk["model_factors"] = factors
    return {
        "case_id": case_id, "case_type": case_type.name, "facts": facts, "sources": sources, "errors": errors,
        "mismatches": mismatches, "coverage": coverage, "red_flags": flags,
        "similar_past_cases": similar_cases, "risk": risk,
    }


def main_issue(findings: dict[str, Any]) -> str:
    """One line for the case list."""
    if findings["mismatches"]:
        return findings["mismatches"][0]["description"]
    if findings["coverage"]["status"] == "excluded":
        return "The policy appears to exclude this loss."
    if findings["red_flags"]:
        return findings["red_flags"][0]["title"]
    return "No issue found"
