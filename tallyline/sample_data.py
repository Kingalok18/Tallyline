"""Build the sample cases, past cases, knowledge texts and answer key.

All wording specific to one kind of insurance comes from the case type's sample_content
folder. The answer key is read only by evaluate.py and the tests, never by the app.
"""
from __future__ import annotations

import json
import random
import re
import shutil
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

import yaml

from . import checks, db, llm, risk_model
from .case_types import CaseType, load_case_types
from .settings import Settings

SAMPLE_LINE = "SAMPLE TEXT FOR A STUDENT PROJECT – NOT A REAL POLICY."
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

CASE_PREFIX = "TL-"

# Which warning sign in the general guide each planted mismatch should trigger.
MISMATCH_FLAGS = {
    "date_order": "RF-G06", "early_incident": "RF-G01", "loss_detail": "RF-G07", "billed_items": "RF-G05",
    "invoice_total": "RF-G04", "incident_report": "RF-G08", "provider_repeat": "RF-G09",
}
MISMATCH_TYPES = list(MISMATCH_FLAGS)
UNANSWERABLE = [
    "What is the policyholder's bank account number?",
    "What is the policyholder's blood group?",
    "Which school did the policyholder attend?",
    "What is the name of the policyholder's employer?",
    "What is the policyholder's email address?",
]
NARRATIVE_INSTRUCTIONS = {
    "statement": "the policyholder's account of what happened, in the first person, in 5 to 7 sentences.",
    "assessment_report": ("the assessor's observations from the inspection, in the third person, in 3 to 4 sentences. "
                          "Refer to the policyholder by that word and never write I or my. Do not list items or amounts."),
    "incident_report": ("an official's short record of the incident as the policyholder reported it, in the third person, "
                        "in 2 to 3 sentences. Refer to the policyholder by that word and never write I or my."),
}
NARRATIVE_WORDS = {"statement": (30, 200), "assessment_report": (25, 130), "incident_report": (20, 120)}
_FIRST_PERSON = re.compile(r"\b(I|my|me|mine)\b")


@dataclass
class OpenCase:
    """Everything known about one sample case while it is being built."""

    case_id: str
    row: dict[str, Any]
    is_fraud: bool
    planted: str | None
    incident_date: date
    policy_start: date
    invoice_date: date
    location: str
    description: str
    coverage_status: str
    coverage_clauses: list[str]
    loss_detail_statement: str
    loss_detail_assessment: str
    items_assessed: dict[str, float]
    items_billed: dict[str, float]
    estimate: float
    invoice_total: float
    provider: str = ""
    statement_says_filed: bool = False
    report_exists: bool = False
    cue_sentences: list[str] = field(default_factory=list)
    flags: set[str] = field(default_factory=set)
    policy_number: str = ""
    lookalike_id: str | None = None


# ----- building the facts -----


def _incident_date(row: dict[str, Any], mapping: dict[str, str], rng: random.Random) -> date:
    """Turn the dataset's year, month, week-of-month and weekday columns into a date."""
    year = int(row[mapping["year"]])
    month = MONTHS.index(str(row[mapping["month"]])) + 1
    week = int(row[mapping["week_of_month"]])
    weekday = WEEKDAYS.index(str(row[mapping["day_of_week"]]))
    first = date(year, month, 1)
    day = first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (week - 1))
    while day.month != month:
        day -= timedelta(days=7)
    return day


def _prices(items: list[str], low: float, high: float, rng: random.Random) -> dict[str, float]:
    return {item: float(rng.randint(int(low), int(high))) for item in items}


def _money(value: float, currency: str) -> str:
    return f"{currency} {value:.2f}"


def _plan_mismatches(rows: list[dict[str, Any]], label: str, settings: Settings, rng: random.Random) -> list[str | None]:
    """Decide which cases get a mismatch; types rotate so each gets used."""
    order = MISMATCH_TYPES[:]
    rng.shuffle(order)
    plan: list[str | None] = []
    next_type = 0
    for row in rows:
        rate = settings.mismatch_rate_fraud if int(row[label]) else settings.mismatch_rate_genuine
        if rng.random() < rate:
            plan.append(order[next_type % len(order)])
            next_type += 1
        else:
            plan.append(None)
    return plan


def _coverage(scenario: dict[str, Any], settings: Settings, rng: random.Random) -> tuple[str, list[str], str, str | None]:
    """Choose the coverage situation: (status, clause ids, description, extra item)."""
    draw = rng.random()
    if draw < settings.coverage_rate_excluded:
        chosen = rng.choice(scenario["exclusions"])
        return "excluded", [chosen["clause"]], chosen["description"], None
    if draw < settings.coverage_rate_excluded + settings.coverage_rate_partly:
        chosen = rng.choice(scenario["partly_covered"])
        return "partly", [chosen["clause"], chosen["limit_clause"]], chosen["description"], chosen["extra_item"]
    chosen = rng.choice(scenario["covered_incidents"])
    return "covered", [chosen["clause"]], chosen["description"], None


def _build_case(
    index: int, row: dict[str, Any], planted: str | None, label: str, scenario: dict[str, Any],
    settings: Settings, rng: random.Random,
) -> OpenCase:
    is_fraud = bool(int(row[label]))
    incident = _incident_date(row, scenario["dataset_mapping"], rng)
    status, clauses, description, extra_item = _coverage(scenario, settings, rng)

    details = sorted(scenario["loss_details"])
    loss_detail = rng.choice(details)
    pool = list(scenario["loss_details"][loss_detail])
    count = rng.randint(min(3, len(pool)), len(pool))
    low, high = scenario["item_price_range"]
    assessed = _prices(rng.sample(pool, count), low, high, rng)
    if extra_item:
        assessed[extra_item] = float(rng.randint(int(low), int(high)))
    estimate = sum(assessed.values())
    factor = rng.uniform(0.92, 1.1)
    billed = {item: round(price * factor, 0) for item, price in assessed.items()}

    policy_start = incident - timedelta(days=rng.randint(settings.early_incident_days + 31, 700))
    invoice_date = incident + timedelta(days=rng.randint(3, 25))
    assessment_detail = loss_detail
    report_exists = rng.random() < scenario["incident_report_rate"]
    says_filed = report_exists

    if planted == "date_order":
        invoice_date = incident - timedelta(days=rng.randint(2, 10))
    elif planted == "early_incident":
        policy_start = incident - timedelta(days=rng.randint(1, settings.early_incident_days - 1))
    elif planted == "loss_detail":
        assessment_detail = rng.choice([d for d in details if not set(d.split()) & set(loss_detail.split())])
    elif planted == "billed_items":
        billed[rng.choice(scenario["extra_billed_items"])] = float(rng.randint(int(low), int(high)))
    elif planted == "invoice_total":
        ratio = rng.uniform(1.4, 1.7)
        billed = {item: round(price * ratio, 0) for item, price in assessed.items()}
    elif planted == "incident_report":
        report_exists, says_filed = False, True

    case = OpenCase(
        case_id=f"{CASE_PREFIX}{index:04d}", row=row, is_fraud=is_fraud, planted=planted, incident_date=incident,
        policy_start=policy_start, invoice_date=invoice_date, location=rng.choice(scenario["cities"]),
        description=description, coverage_status=status, coverage_clauses=clauses,
        loss_detail_statement=loss_detail, loss_detail_assessment=assessment_detail,
        items_assessed=assessed, items_billed=billed, estimate=estimate,
        invoice_total=sum(billed.values()), statement_says_filed=says_filed, report_exists=report_exists,
        policy_number=f"PN-{rng.randint(100000, 999999)}",
    )
    if planted:
        case.flags.add(MISMATCH_FLAGS[planted])
    _add_cues(case, scenario, rng)
    return case


def _add_cues(case: OpenCase, scenario: dict[str, Any], rng: random.Random) -> None:
    for cue in scenario.get("cues", []):
        if rng.random() < (cue["rate_fraud"] if case.is_fraud else cue["rate_genuine"]):
            case.cue_sentences.append(cue["sentence"])
            case.flags.add(cue["flag_id"])
    for cue in scenario.get("row_cues", []):
        if str(case.row.get(cue["column"])) in cue["values"]:
            case.cue_sentences.append(cue["sentence"])
            case.flags.add(cue["flag_id"])


# ----- writing documents -----


def _context(case: OpenCase, loss_detail: str) -> dict[str, str]:
    return {
        "incident_date": case.incident_date.isoformat(), "location": case.location,
        "incident_description": case.description, "loss_detail": loss_detail, "provider": case.provider,
    }


NarrativeWriter = Callable[[str, str, dict[str, str], list[str], str], str]


def template_writer(doc: str, label: str, facts: dict[str, str], required: list[str], template: str) -> str:
    """Write a narrative by filling the scenario's template. Needs no language model."""
    return template.format(**facts)


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+", text))


def _narrative_ok(doc: str, text: str, facts: dict[str, str], required: list[str]) -> bool:
    """True if the text has the required facts, a sensible length and no invented numbers."""
    low, high = NARRATIVE_WORDS[doc]
    if not low <= len(text.split()) <= high or "\n#" in text:
        return False
    if any(value.lower() not in text.lower() for value in required):
        return False
    if doc != "statement" and _FIRST_PERSON.search(text):
        return False
    return _numbers(text) <= _numbers(" ".join(facts.values()))


def make_llm_writer(settings: Settings) -> NarrativeWriter:
    """Write a narrative with the language model, using the template if the output fails its checks."""
    prompt_text = settings.prompt("write_documents")

    def write(doc: str, label: str, facts: dict[str, str], required: list[str], template: str) -> str:
        fact_lines = "\n".join(f"- {key}: {value}" for key, value in facts.items())
        prompt = prompt_text.format(document_label=label, instructions=NARRATIVE_INSTRUCTIONS[doc], facts=fact_lines)
        for attempt in range(2):
            text = llm.complete(prompt, settings, max_tokens=320, attempt=attempt).strip()
            if _narrative_ok(doc, text, facts, required):
                return text
        return template.format(**facts)

    return write


def _header(label: str, case: OpenCase) -> list[str]:
    return [
        f"# {label}", "", f"Case: {case.case_id}", "", f"Policy number: {case.policy_number}", "",
        f"Policy start date: {case.policy_start.isoformat()}", "",
    ]


def write_case_documents(
    case: OpenCase, case_type: CaseType, scenario: dict[str, Any], writer: NarrativeWriter, rng: random.Random
) -> dict[str, str]:
    """Return the Markdown for each document of this case."""
    terms, labels, currency = case_type.terms, case_type.documents, scenario["currency"]
    statement_facts = _context(case, case.loss_detail_statement)
    assessment_facts = _context(case, case.loss_detail_assessment)
    report_label = labels.get("incident_report", "Incident report")

    narrative = writer(
        "statement", labels["statement"], statement_facts,
        [statement_facts["incident_date"], case.location, case.loss_detail_statement],
        rng.choice(scenario["statement_templates"]),
    )
    statement = _header(labels["statement"], case) + [" ".join([narrative, *case.cue_sentences]), ""] + [
        "## Declared details", "",
        f"- Incident date: {statement_facts['incident_date']}",
        f"- Location: {case.location}",
        f"- Incident description: {case.description}",
        f"- {terms['loss_detail']}: {case.loss_detail_statement}",
        f"- {terms['provider']}: {case.provider}",
        f"- {report_label} filed: {'Yes' if case.statement_says_filed else 'No'}",
    ]
    observations = writer(
        "assessment_report", labels["assessment_report"], assessment_facts,
        [case.loss_detail_assessment], rng.choice(scenario["assessment_templates"]),
    )
    assessment = _header(labels["assessment_report"], case) + [observations, ""] + [
        f"## {terms['loss_detail']}", "", case.loss_detail_assessment, "",
        f"## {terms['items']} assessed", "", *[f"- {item}" for item in case.items_assessed], "",
        "## Estimate", "", _money(case.estimate, currency),
    ]
    invoice = _header(labels["invoice"], case) + [
        f"{terms['provider']}: {case.provider}", "", f"Invoice date: {case.invoice_date.isoformat()}", "",
        "| Item | Charge |", "|---|---|",
        *[f"| {item} | {_money(price, currency)} |" for item, price in case.items_billed.items()],
        f"| Total | {_money(case.invoice_total, currency)} |",
    ]
    documents = {
        "statement": "\n".join(statement) + "\n",
        "assessment_report": "\n".join(assessment) + "\n",
        "invoice": "\n".join(invoice) + "\n",
    }
    if case.report_exists and "incident_report" in labels:
        record = writer(
            "incident_report", report_label, statement_facts, [statement_facts["incident_date"], case.location],
            rng.choice(scenario["incident_report_templates"]),
        )
        documents["incident_report"] = "\n".join(_header(report_label, case) + [
            record, "", "## Details", "",
            f"- Incident date: {statement_facts['incident_date']}", f"- Location: {case.location}",
        ]) + "\n"
    return documents


# ----- past cases -----


def _past_outcome(is_fraud: bool, rng: random.Random) -> str:
    if is_fraud:
        return rng.choice(["rejected", "sent_for_investigation"])
    return "paid" if rng.random() < 0.85 else "paid_reduced"


OUTCOME_NOTES = {
    "paid": "The documents agreed with each other and the loss fell within cover. The case was paid in full.",
    "paid_reduced": "The loss fell within cover but part of the amount was not supported. The case was paid at a reduced amount.",
    "rejected": "The documents did not support the account given. The case was rejected.",
    "sent_for_investigation": "The documents raised unanswered questions. The case was sent for investigation.",
}


def _past_case(
    past_id: str, row: dict[str, Any], label: str, case_type: CaseType, scenario: dict[str, Any],
    providers: list[str], rng: random.Random,
) -> dict[str, Any]:
    is_fraud = bool(int(row[label]))
    description = rng.choice(scenario["covered_incidents"] + scenario["exclusions"])["description"]
    return {
        "past_case_id": past_id, "case_type": case_type.name, "sample": True,
        "incident_date": _incident_date(row, scenario["dataset_mapping"], rng).isoformat(),
        "location": rng.choice(scenario["cities"]), "description": description,
        "provider": rng.choice(providers), "amount": float(rng.randint(*scenario["past_amount_range"])),
        "currency": scenario["currency"], "outcome": _past_outcome(is_fraud, rng),
    }


def _finish_past_case(record: dict[str, Any], case_type: CaseType) -> dict[str, Any]:
    """Write the summary and the 3-sentence note once all fields are final."""
    provider_term = case_type.terms["provider"]
    record["summary"] = (
        f"Incident on {record['incident_date']} in {record['location']}: {record['description']}. "
        f"{provider_term}: {record['provider']}. Amount: {_money(record['amount'], record['currency'])}."
    )
    record["note"] = (
        f"The case concerned an incident in {record['location']} handled by {record['provider']}. "
        + OUTCOME_NOTES[record["outcome"]]
    )
    return record


def _plant_past_links(
    cases: list[OpenCase], past: list[dict[str, Any]], settings: Settings, rng: random.Random
) -> None:
    """Add look-alike past cases and the recent past cases behind a repeated provider."""
    free = list(range(len(past)))
    rng.shuffle(free)
    for case in cases:
        if case.planted != "provider_repeat":
            continue
        for _ in range(settings.provider_repeat_threshold + 1):
            record = past[free.pop()]
            record["provider"] = case.provider
            days = rng.randint(1, settings.provider_recent_days - 1)
            record["incident_date"] = (case.incident_date - timedelta(days=days)).isoformat()
    n_lookalikes = min(round(len(cases) * settings.lookalike_share), len(free))
    for case in rng.sample(cases, n_lookalikes):
        record = past[free.pop()]
        record["provider"] = case.provider
        record["description"] = case.description
        record["location"] = case.location
        record["amount"] = round(case.invoice_total * rng.uniform(0.85, 1.15), 0)
        days = rng.randint(settings.provider_recent_days + 30, 500)
        record["incident_date"] = (case.incident_date - timedelta(days=days)).isoformat()
        case.lookalike_id = record["past_case_id"]


# ----- answer key -----


def _gold_facts(case: OpenCase) -> dict[str, dict[str, Any]]:
    """The facts each document states; used to score extraction and to confirm planted mismatches."""
    facts: dict[str, dict[str, Any]] = {
        "statement": {
            "incident_date": case.incident_date.isoformat(), "location": case.location,
            "incident_description": case.description, "loss_detail": case.loss_detail_statement,
            "provider": case.provider, "incident_report_filed": case.statement_says_filed,
        },
        "assessment_report": {
            "loss_detail": case.loss_detail_assessment, "items_assessed": list(case.items_assessed),
            "estimate": case.estimate,
        },
        "invoice": {
            "items_billed": list(case.items_billed), "invoice_total": case.invoice_total,
            "invoice_date": case.invoice_date.isoformat(), "provider": case.provider,
        },
    }
    if case.report_exists:
        facts["incident_report"] = {"incident_date": case.incident_date.isoformat(), "location": case.location}
    return facts


def _mismatch_record(kind: str, case: OpenCase, case_type: CaseType, scenario: dict[str, Any]) -> dict[str, Any]:
    """Name the documents and the exact text that show a mismatch."""
    currency = scenario["currency"]
    terms = case_type.terms
    report_label = case_type.documents.get("incident_report", "Incident report")
    extra = [item for item in case.items_billed if item not in case.items_assessed]
    evidence = {
        "date_order": [("statement", f"Incident date: {case.incident_date}"), ("invoice", f"Invoice date: {case.invoice_date}")],
        "early_incident": [("statement", f"Policy start date: {case.policy_start}"), ("statement", f"Incident date: {case.incident_date}")],
        "loss_detail": [("statement", f"{terms['loss_detail']}: {case.loss_detail_statement}"), ("assessment_report", case.loss_detail_assessment)],
        "billed_items": [("invoice", extra[0] if extra else ""), ("assessment_report", f"{terms['items']} assessed")],
        "invoice_total": [("invoice", _money(case.invoice_total, currency)), ("assessment_report", _money(case.estimate, currency))],
        "incident_report": [("statement", f"{report_label} filed: Yes")],
        "provider_repeat": [("invoice", f"{terms['provider']}: {case.provider}")],
    }[kind]
    return {"type": kind, "documents": sorted({doc for doc, _ in evidence}),
            "evidence": [{"document": doc, "text": text} for doc, text in evidence]}


def _questions(case: OpenCase, case_type: CaseType, scenario: dict[str, Any], rng: random.Random) -> tuple[list[dict[str, str]], str]:
    currency = scenario["currency"]
    provider_term = case_type.terms["provider"].lower()
    pool = [
        {"question": "What is the total amount on the invoice?", "answer": f"{case.invoice_total:.2f}", "document": "invoice", "shown_as": _money(case.invoice_total, currency)},
        {"question": "What estimate is given in the assessment report?", "answer": f"{case.estimate:.2f}", "document": "assessment_report", "shown_as": _money(case.estimate, currency)},
        {"question": f"Which {provider_term} issued the invoice?", "answer": case.provider, "document": "invoice", "shown_as": case.provider},
        {"question": "On what date did the incident happen according to the policyholder's statement?", "answer": case.incident_date.isoformat(), "document": "statement", "shown_as": case.incident_date.isoformat()},
        {"question": "In which place did the incident happen?", "answer": case.location, "document": "statement", "shown_as": case.location},
        {"question": "What is the policy number?", "answer": case.policy_number, "document": "statement", "shown_as": case.policy_number},
        {"question": "What is the date of the invoice?", "answer": case.invoice_date.isoformat(), "document": "invoice", "shown_as": case.invoice_date.isoformat()},
    ]
    return rng.sample(pool, 4), rng.choice(UNANSWERABLE)


# ----- the whole build -----


def _copy_knowledge(case_type: CaseType, settings: Settings) -> None:
    """Copy the sample texts into knowledge/ unless the folder already has the user's documents."""
    content = settings.root / case_type.sample_content
    targets = [
        (content.parent / "general_red_flags.md", settings.knowledge_dir / "general" / "red_flags" / "sample_red_flags.md"),
        (content / "policy.md", settings.root / case_type.knowledge_folder / "policy" / "sample_policy.md"),
        (content / "red_flags.md", settings.root / case_type.knowledge_folder / "red_flags" / "sample_red_flags.md"),
    ]
    for source, target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        others = [p for p in target.parent.iterdir() if p.name != target.name]
        if not others:
            shutil.copyfile(source, target)


def _remove_old_sample_cases(settings: Settings) -> None:
    """Delete generated cases and their stored analyses; hand-added cases stay."""
    for path in sorted(settings.cases_dir.glob("*/case.json")):
        if json.loads(path.read_text(encoding="utf-8")).get("sample"):
            shutil.rmtree(path.parent)
    db.execute(settings, "DELETE FROM analyses WHERE case_id IN (SELECT id FROM cases WHERE is_sample = 1)")
    db.execute(settings, "DELETE FROM asks WHERE case_id IN (SELECT id FROM cases WHERE is_sample = 1)")
    db.execute(settings, "DELETE FROM decisions WHERE case_id IN (SELECT id FROM cases WHERE is_sample = 1)")
    db.execute(settings, "DELETE FROM citation_log WHERE case_id IN (SELECT id FROM cases WHERE is_sample = 1)")
    # Rebuilt cases reuse the same IDs, so they must not inherit the old status or risk score.
    db.execute(settings, "DELETE FROM cases WHERE is_sample = 1")


def _safe_features(row: dict[str, Any], case_type: CaseType) -> dict[str, Any]:
    if not case_type.risk_model:
        return {}  # generated rows are not data, and there is no model to read them
    excluded = risk_model.excluded_columns(case_type)
    return {key: value for key, value in row.items() if key not in excluded}


SAMPLE_LABEL = "sample_fraud_label"


def _rows_for(
    case_type: CaseType, scenario: dict[str, Any], settings: Settings, n_open: int, n_past: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, bool]:
    """Rows that seed the open and past cases: (open, past, label column, is real data).

    A type with a dataset uses its real rows. A type without one gets generated rows from
    `sample_columns`, with a made-up fraud label at `sample_fraud_rate`.
    """
    if case_type.risk_model:
        rows, is_real = risk_model.load_dataset(case_type, settings)
        _, open_df, past_df = risk_model.split_rows(rows, case_type, settings, n_open, n_past)
        return open_df.to_dict("records"), past_df.to_dict("records"), case_type.risk_model["label_column"], is_real
    rng = random.Random(f"{settings.seed}:{case_type.name}:rows")

    def row() -> dict[str, Any]:
        values = {name: rng.choice(options) for name, options in scenario["sample_columns"].items()}
        values[SAMPLE_LABEL] = int(rng.random() < scenario["sample_fraud_rate"])
        return values

    return [row() for _ in range(n_open)], [row() for _ in range(n_past)], SAMPLE_LABEL, False


def _build_type(
    case_type: CaseType, settings: Settings, writer: NarrativeWriter, first_id: int, first_past: int, n_open: int, n_past: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    rng = random.Random(f"{settings.seed}:{case_type.name}")
    content = settings.root / case_type.sample_content
    scenario = yaml.safe_load((content / "scenario.yaml").read_text(encoding="utf-8"))
    open_rows, past_rows, label, is_real = _rows_for(case_type, scenario, settings, n_open, n_past)
    _copy_knowledge(case_type, settings)

    plan = _plan_mismatches(open_rows, label, settings, rng)
    cases = [
        _build_case(first_id + i, row, plan[i], label, scenario, settings, rng)
        for i, row in enumerate(open_rows)
    ]
    providers = list(scenario["providers"])
    rng.shuffle(providers)
    n_reserved = sum(1 for case in cases if case.planted == "provider_repeat")
    reserved, general_pool = providers[:n_reserved], providers[n_reserved:]
    for case in cases:
        case.provider = reserved.pop() if case.planted == "provider_repeat" else rng.choice(general_pool)

    past = [
        _past_case(f"PC-{first_past + i:04d}", row, label, case_type, scenario, general_pool, rng)
        for i, row in enumerate(past_rows)
    ]
    _plant_past_links(cases, past, settings, rng)
    past = [_finish_past_case(record, case_type) for record in past]

    key_rows = []
    for number, case in enumerate(cases, start=1):
        documents = write_case_documents(case, case_type, scenario, writer, rng)
        folder = settings.cases_dir / case.case_id
        folder.mkdir(parents=True, exist_ok=True)
        for doc, text in documents.items():
            (folder / f"{doc}.md").write_text(text, encoding="utf-8")
        record = {
            "case_id": case.case_id, "case_type": case_type.name, "sample": True,
            "policy_number": case.policy_number, "policy_start": case.policy_start.isoformat(),
            "incident_date": case.incident_date.isoformat(), "amount": case.invoice_total,
            "currency": scenario["currency"], "provider": case.provider,
            "features": _safe_features(case.row, case_type),
        }
        (folder / "case.json").write_text(json.dumps(record, indent=2, sort_keys=True, default=str), encoding="utf-8")
        key_rows.append(_key_row(case, case_type, scenario, past, settings, rng))
        print(f"  documents written: {case.case_id} ({number}/{len(cases)})", flush=True)
    return key_rows, past, is_real


def _key_row(
    case: OpenCase, case_type: CaseType, scenario: dict[str, Any], past: list[dict[str, Any]],
    settings: Settings, rng: random.Random,
) -> dict[str, Any]:
    facts = _gold_facts(case)
    found = checks.run_rules(
        facts, {}, case.policy_start.isoformat(), set(facts), past, settings, case_type.name
    )
    kinds = {m["type"] for m in found}
    flags = set(case.flags) | {MISMATCH_FLAGS[kind] for kind in kinds}
    questions, unanswerable = _questions(case, case_type, scenario, rng)
    return {
        "case_id": case.case_id, "case_type": case_type.name, "fraud_label": int(case.is_fraud),
        "coverage_status": case.coverage_status, "coverage_clauses": case.coverage_clauses,
        "expected_red_flags": sorted(flags),
        "planted_mismatches": [_mismatch_record(kind, case, case_type, scenario) for kind in sorted(kinds)],
        "lookalike_past_case": case.lookalike_id, "questions": questions, "unanswerable_question": unanswerable,
        "facts": facts,
    }


def resolve_writer(settings: Settings, writer: str | None) -> tuple[NarrativeWriter, str]:
    """Pick the narrative writer, falling back to templates if the model is unavailable."""
    choice = writer or settings.document_writer
    if choice == "template":
        return template_writer, "template"
    status = llm.check_ollama(settings)
    if status["reachable"] and status["model_present"]:
        return make_llm_writer(settings), "llm"
    print(f"Language model {settings.llm_model} is not available; writing documents from templates.")
    return template_writer, "template (model unavailable)"


def build_sample_data(settings: Settings, writer: str | None = None) -> str:
    """Build cases, past cases, knowledge and the answer key for every type with sample content."""
    types = [t for t in load_case_types(settings).values() if t.sample_content]
    if not types:
        return "No case type has sample content; nothing to build."
    narrative_writer, writer_name = resolve_writer(settings, writer)
    counts = risk_model.sample_counts(settings)
    settings.cases_dir.mkdir(parents=True, exist_ok=True)
    _remove_old_sample_cases(settings)
    all_keys: list[dict[str, Any]] = []
    all_past: list[dict[str, Any]] = []
    real_flags: dict[str, bool] = {}
    for case_type in types:
        keys, past, is_real = _build_type(
            case_type, settings, narrative_writer, len(all_keys) + 1, len(all_past) + 1, *counts[case_type.name]
        )
        all_keys += keys
        all_past += past
        real_flags[case_type.name] = is_real
    _write_jsonl(settings.past_cases_path, all_past)
    _write_jsonl(settings.answer_key_path, all_keys)
    (settings.data_dir / "sample_info.json").write_text(
        json.dumps({"writer": writer_name, "real_dataset": real_flags}, indent=2), encoding="utf-8"
    )
    return f"Built {len(all_keys)} open cases and {len(all_past)} past cases (writer: {writer_name}; real dataset: {real_flags})."


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
