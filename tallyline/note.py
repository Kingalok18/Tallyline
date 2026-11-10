"""Investigator note, "Ask the documents", and the citation check that runs on both."""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Callable, Iterable

from . import NOT_FOUND, checks, db, llm
from .settings import Settings

ACTIONS = ("pay", "ask_for_documents", "investigate", "reject")
_ID_RE = re.compile(r"\b[A-Z]{1,4}-[A-Z]?\d+\b")
_NUMBER_RE = re.compile(r"\d{4}-\d{2}-\d{2}|\d+(?:,\d{3})*(?:\.\d+)?")
_LABEL_RE = re.compile(r"\bP\d+\b")


def number_tokens(text: str) -> set[str]:
    """Numbers, dates and reference IDs in a text, in a comparable form."""
    tokens = set(_ID_RE.findall(text))
    for match in _NUMBER_RE.findall(_ID_RE.sub(" ", text)):
        if "-" in match:
            tokens.add(match)
        else:
            tokens.add(str(float(match.replace(",", ""))))
    return tokens


def check_citations(
    sentences: Iterable[dict[str, Any]], given_passages: dict[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Keep sentences whose citations are valid and whose numbers appear in the cited passages.

    Each sentence is {"text", "passages": [ids]}; `given_passages` maps every passage ID
    given to the model to its text. Returns (kept, removed), where each removed entry
    names the rule the sentence failed.
    """
    kept, removed = [], []
    for sentence in sentences:
        text, cited = str(sentence.get("text", "")).strip(), list(sentence.get("passages", []))
        if not text:
            continue
        if not cited:
            removed.append({"sentence": text, "reason": "no citation"})
        elif any(pid not in given_passages for pid in cited):
            removed.append({"sentence": text, "reason": "cites a passage that was not given"})
        else:
            cited_text = " ".join(given_passages[pid] for pid in cited)
            unsupported = sorted(number_tokens(text) - number_tokens(cited_text))
            if unsupported:
                removed.append({"sentence": text, "reason": f"number or date not in the cited passages: {unsupported}"})
            else:
                kept.append({"text": text, "passages": cited})
    return kept, removed


def _passage_texts(ids: Iterable[str], settings: Settings) -> dict[str, str]:
    ids = list(dict.fromkeys(i for i in ids if i))
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    rows = db.query(settings, f"SELECT id, text FROM passages WHERE id IN ({marks})", ids)
    return {row["id"]: row["text"] for row in rows}


def _note_passage_ids(findings: dict[str, Any]) -> list[str]:
    ids: list[str] = list(findings["coverage"]["passages"])
    for mismatch in findings["mismatches"]:
        ids += mismatch["passages"]
    for flag in findings["red_flags"]:
        ids += [flag["guide_passage"], flag["case_passage"]]
    ids += [past["passage"] for past in findings["similar_past_cases"][:3]]
    for doc_sources in findings["sources"].values():
        ids += list(doc_sources.values())[:1]
    return list(dict.fromkeys(i for i in ids if i))[:12]


def _findings_text(findings: dict[str, Any], labels: dict[str, str]) -> str:
    """Summarise the findings for the note prompt, using short passage labels."""

    def refs(ids: Iterable[str | None]) -> str:
        return " ".join(f"[{labels[i]}]" for i in ids if i in labels)

    def name(flag: dict[str, Any]) -> str:
        return flag["title"].removeprefix(str(flag["flag_id"])).strip()

    # ID codes stay out of this text: the model repeats them and then cites the wrong passage.
    documents = refs(list(doc_sources.values())[0] for doc_sources in findings["sources"].values() if doc_sources)
    coverage = findings["coverage"]
    lines = [f"Coverage: {coverage['status']}. {coverage['reason']} {refs(coverage['passages'])}"]
    lines += [f"Mismatch: {m['description']} {refs(m['passages'])}" for m in findings["mismatches"]] or [f"Mismatches: none {documents}"]
    lines += [f"Warning sign: {name(f)}. The case says: {f['evidence']} {refs([f['guide_passage'], f['case_passage']])}"
              for f in findings["red_flags"]] or [f"Warning signs: none {documents}"]
    past = findings["similar_past_cases"][:3]
    if past:
        outcomes = Counter(p["outcome"].replace("_", " ") for p in past)
        summary = ", ".join(f"{count} {outcome}" for outcome, count in outcomes.items())
        lines.append(f"Similar past cases: {len(past)} found, with these outcomes: {summary}. {refs(p['passage'] for p in past)}")
    else:
        lines.append(f"Similar past cases: none {documents}")
    return "\n".join(f"{number}. {line}" for number, line in enumerate(lines, start=1))


def recommended_action(findings: dict[str, Any]) -> str:
    """The action follows from the findings, so it is decided here and not by the model."""
    coverage = findings["coverage"]["status"]
    if coverage == "excluded":
        return "reject"
    if findings["mismatches"] or findings["red_flags"]:
        return "investigate"
    if coverage == "unclear" or findings["errors"]:
        return "ask_for_documents"
    return "pay"


def _clean_sentence(text: Any) -> tuple[str, list[str]]:
    """Take passage labels the model wrote into a sentence out of its text; they count as citations."""
    raw = str(text)
    labels = _LABEL_RE.findall(raw)
    cleaned = _LABEL_RE.sub("", re.sub(r"[\[(]\s*(?:P\d+[\s,;]*(?:and\s+)?)+\s*[\])]", " ", raw))
    cleaned = re.sub(r"\s+([.,;:])", r"\1", " ".join(cleaned.split()))
    return cleaned, labels


def write_note(findings: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """One model call that writes the note, followed by the citation check."""
    texts = _passage_texts(_note_passage_ids(findings), settings)
    labels = {pid: f"P{number}" for number, pid in enumerate(texts, start=1)}
    aliases = {label: pid for pid, label in labels.items()}
    # Short on purpose: given long passages, the model summarises them instead of the findings.
    # The citation check still uses each passage's full text.
    block = "\n\n".join(f"[{labels[pid]}] {text[:250]}" for pid, text in texts.items())
    summary = _findings_text(findings, labels)
    count = summary.count("\n") + 1
    prompt = settings.prompt("note").format(findings=summary, passages=block, count=count)
    try:
        value = llm.complete_json(prompt, settings, required_keys=["sentences"], max_tokens=450)
    except llm.LLMError as exc:
        return {"sentences": [], "action": None, "removed": [], "error": str(exc), "total": 0}
    raw = value.get("sentences") if isinstance(value.get("sentences"), list) else []
    sentences, written = [], set()
    for entry in raw:
        if isinstance(entry, dict):
            text, in_text = _clean_sentence(entry.get("text", ""))
            if text.lower() in written:  # the model sometimes repeats itself
                continue
            written.add(text.lower())
            cited = [c.strip("[] ") for c in checks.as_list(entry.get("passages"))] + in_text
            sentences.append({"text": text, "passages": list(dict.fromkeys(aliases.get(c, c) for c in cited))})
    sentences = sentences[:count]
    kept, removed = check_citations(sentences, texts)
    for item in removed:
        db.log_removal(settings, findings["case_id"], "note", item["sentence"], item["reason"])
    return {"sentences": kept, "action": recommended_action(findings), "removed": removed, "total": len(sentences)}


def full_analysis(
    case_id: str, settings: Settings, progress: Callable[[str], None] = lambda _: None
) -> dict[str, Any]:
    """Run all checks, write the note and save the result."""
    findings = checks.run_checks(case_id, settings, progress)
    progress("Writing the note")
    findings["note"] = write_note(findings, settings)
    db.execute(
        settings, "INSERT OR REPLACE INTO analyses (case_id, result_json, created_at) VALUES (?, ?, ?)",
        (case_id, json.dumps(findings), db.now()),
    )
    db.execute(
        settings,
        "UPDATE cases SET risk_score = ?, risk_level = ?, main_issue = ?, status = CASE WHEN status = 'Open' THEN 'Analysed' ELSE status END WHERE id = ?",
        (findings["risk"]["score"], findings["risk"]["level"], checks.main_issue(findings), case_id),
    )
    return findings


def ask_documents(
    case_id: str, question: str, settings: Settings, include_knowledge: bool = False, save: bool = True
) -> dict[str, Any]:
    """Answer from retrieved passages only, with citations, or say it was not found."""
    from . import rag

    case_type = checks.load_case(case_id, settings)["case_type"]
    passages = rag.retrieve("case_docs", question, {"case_id": case_id}, settings=settings, min_score=rag.NO_MIN_SCORE)
    if include_knowledge:
        passages += rag.retrieve("knowledge", question, {"case_type": [case_type.name, "general"]}, settings=settings)
        passages += rag.retrieve("past_cases", question, {"case_type": case_type.name}, settings=settings)
    result = _answer(question, passages, case_id, settings)
    if save:
        db.execute(settings, "INSERT INTO asks (case_id, question, answer_json, created_at) VALUES (?, ?, ?, ?)",
                   (case_id, question, json.dumps(result), db.now()))
    return result


def _answer(question: str, passages: list[Any], case_id: str, settings: Settings) -> dict[str, Any]:
    not_found = {"question": question, "answer": NOT_FOUND, "passages": [], "removed": []}
    if not passages:
        return not_found
    labels = {f"P{number}": passage for number, passage in enumerate(passages, start=1)}
    # A 300-token chunk can run to about 1500 characters, and the answer may be at its end.
    block = "\n\n".join(f"[{label}] {passage.text[:1600]}" for label, passage in labels.items())
    prompt = settings.prompt("ask").format(passages=block, question=question)
    try:
        value = llm.complete_json(prompt, settings, required_keys=["answer"], max_tokens=150)
    except llm.LLMError as exc:
        return {**not_found, "error": str(exc)}
    answer = str(value.get("answer", "")).strip()
    if not answer or "not found in the documents" in answer.lower():
        return not_found
    cited = [c.strip("[] ") for c in checks.as_list(value.get("passages"))]
    ids = [labels[c].id if c in labels else c for c in cited]
    kept, removed = check_citations([{"text": answer, "passages": ids}], {p.id: p.text for p in passages})
    for item in removed:
        db.log_removal(settings, case_id, "ask", item["sentence"], item["reason"])
    if not kept:
        return {**not_found, "removed": removed}
    return {"question": question, "answer": kept[0]["text"], "passages": kept[0]["passages"], "removed": []}
