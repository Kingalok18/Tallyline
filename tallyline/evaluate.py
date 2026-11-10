"""All metrics. Everything is scored against data/answer_key.jsonl and saved to results/.

RESULTS.md is written only from the JSON files in results/, so every number can be traced.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import replace
from typing import Any

from . import NOT_FOUND, checks, db, llm, note, rag, risk_model
from .case_types import load_case_types
from .settings import Settings


def _round(value: float) -> float:
    return round(float(value), 4)


def _share(hits: int, total: int) -> float | None:
    return _round(hits / total) if total else None


def _save(settings: Settings, name: str, payload: dict[str, Any]) -> None:
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    path = settings.results_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved {path}", flush=True)


def _load_json(settings: Settings, name: str) -> dict[str, Any] | None:
    path = settings.results_dir / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _key_rows(settings: Settings, limit: int | None) -> list[dict[str, Any]]:
    rows = db.read_jsonl(settings.answer_key_path)
    return rows[:limit] if limit else rows


def _data_note(settings: Settings) -> dict[str, Any]:
    path = settings.data_dir / "sample_info.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# ----- retrieval (no language model needed) -----


def _retrieval_for(rows: list[dict[str, Any]], settings: Settings) -> dict[str, Any]:
    past = db.query(settings, "SELECT id, case_type, provider, incident_date FROM past_cases")
    clause_hits = clause_total = flag_hits = flag_total = look_hits = look_total = 0
    answer_hits = answer_total = gate_answerable = gate_unanswerable = 0
    for row in rows:
        case_type = row["case_type"]
        statement = row["facts"]["statement"]
        description = statement["incident_description"]
        policy = rag.retrieve("knowledge", description, {"case_type": case_type, "source": "policy"},
                              settings=settings, min_score=rag.NO_MIN_SCORE)
        found_clauses = {p.metadata.get("clause_id") for p in policy}
        for clause in row["coverage_clauses"]:
            clause_total += 1
            clause_hits += clause in found_clauses

        statement_text = (settings.cases_dir / row["case_id"] / "statement.md").read_text(encoding="utf-8")
        mismatches = checks.run_rules(
            row["facts"], {}, _policy_start(row, settings), set(row["facts"]), past, settings, case_type
        )
        queries = checks.red_flag_queries(statement_text, mismatches)
        guide = rag.retrieve_many("knowledge", queries, {"case_type": [case_type, "general"], "source": "red_flags"},
                                  settings, limit=99)
        found_flags = {p.metadata.get("flag_id") for p in guide}
        for flag in row["expected_red_flags"]:
            flag_total += 1
            flag_hits += flag in found_flags

        if row["lookalike_past_case"]:
            question = f"{description} {statement['location']} {statement['provider']}"
            similar = rag.retrieve("past_cases", question, {"case_type": case_type}, settings=settings,
                                   top_n=5, min_score=rag.NO_MIN_SCORE)
            look_total += 1
            look_hits += row["lookalike_past_case"] in {p.metadata.get("past_case_id") for p in similar}

        for item in row["questions"]:
            passages = rag.retrieve("case_docs", item["question"], {"case_id": row["case_id"]},
                                    settings=settings, min_score=rag.NO_MIN_SCORE)
            answer_total += 1
            answer_hits += any(item["answer"].lower() in p.text.lower() for p in passages)
            gate_answerable += bool(rag.retrieve("case_docs", item["question"], {"case_id": row["case_id"]}, settings=settings))
        gate_unanswerable += not rag.retrieve("case_docs", row["unanswerable_question"], {"case_id": row["case_id"]}, settings=settings)
    return {
        "coverage_clause_hit_rate": _share(clause_hits, clause_total), "coverage_clauses": clause_total,
        "red_flag_hit_rate": _share(flag_hits, flag_total), "red_flags": flag_total,
        "lookalike_hit_rate": _share(look_hits, look_total), "lookalikes": look_total,
        "known_answer_hit_rate": _share(answer_hits, answer_total), "known_answers": answer_total,
        "answerable_passing_min_score": _share(gate_answerable, answer_total),
        "unanswerable_stopped_by_min_score": _share(gate_unanswerable, len(rows)), "cases": len(rows),
    }


def _policy_start(row: dict[str, Any], settings: Settings) -> str | None:
    path = settings.cases_dir / row["case_id"] / "case.json"
    return json.loads(path.read_text(encoding="utf-8")).get("policy_start")


def evaluate_retrieval(settings: Settings, limit: int | None = None) -> dict[str, Any]:
    """Hit rates with and without the reranker."""
    rows = _key_rows(settings, limit)
    payload = {
        "data": "sample", "limited": limit is not None, "top_n": settings.top_n_after_rerank, "top_n_past_cases": 5,
        "min_score": settings.min_score,
        "with_reranker": _retrieval_for(rows, replace(settings, use_reranker=True)),
        "without_reranker": _retrieval_for(rows, replace(settings, use_reranker=False)),
    }
    _save(settings, "retrieval", payload)
    return payload


# ----- checks and answers (language model needed) -----


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    return {
        "precision": _share(tp, tp + fp), "recall": _share(tp, tp + fn),
        "true_positives": tp, "false_positives": fp, "false_negatives": fn, "planted": tp + fn,
    }


def _normal(text: Any) -> str:
    return " ".join(str(text).lower().replace(",", "").split())


def _fact_correct(field: str, found: Any, gold: Any) -> bool:
    if field in checks.LIST_FIELDS:
        found_items, gold_items = checks.as_list(found), checks.as_list(gold)
        return len(found_items) == len(gold_items) and all(
            any(checks.similar(a, b) for b in found_items) for a in gold_items
        )
    if field in {"estimate", "invoice_total"}:
        return checks.parse_amount(found) == checks.parse_amount(gold)
    if field == "incident_report_filed":
        return checks.parse_bool(found) == gold
    if field in {"incident_date", "invoice_date"}:
        return checks.parse_date(found) == checks.parse_date(gold)
    if field == "incident_description":
        return checks.similar(str(found or ""), str(gold), threshold=0.5)
    return _normal(gold) == _normal(found)


def evaluate_llm(settings: Settings, limit: int | None = None) -> None:
    """Analyse each case with the language model, then score the checks, note and answers."""
    rows = _key_rows(settings, limit)
    mismatch_counts: dict[str, list[int]] = {}
    flag_tp = flag_fp = flag_fn = coverage_right = 0
    confusion: dict[str, dict[str, int]] = {}
    fact_right = fact_total = 0
    known_right = known_total = unknown_right = 0
    kept = total_sentences = removed_note = removed_ask = 0
    invalid = 0
    started = time.time()
    for number, row in enumerate(rows, start=1):
        findings = note.full_analysis(row["case_id"], settings)
        planted = {m["type"] for m in row["planted_mismatches"]}
        found = {m["type"] for m in findings["mismatches"]}
        for kind in planted | found:
            counts = mismatch_counts.setdefault(kind, [0, 0, 0])
            counts[0] += kind in planted and kind in found
            counts[1] += kind in found and kind not in planted
            counts[2] += kind in planted and kind not in found

        status = findings["coverage"]["status"]
        coverage_right += status == row["coverage_status"]
        line = confusion.setdefault(row["coverage_status"], {})
        line[status] = line.get(status, 0) + 1

        expected = set(row["expected_red_flags"])
        flagged = {flag["flag_id"] for flag in findings["red_flags"]}
        flag_tp += len(expected & flagged)
        flag_fp += len(flagged - expected)
        flag_fn += len(expected - flagged)

        for doc, gold_facts in row["facts"].items():
            for field, gold in gold_facts.items():
                fact_total += 1
                fact_right += _fact_correct(field, findings["facts"].get(doc, {}).get(field), gold)

        note_result = findings["note"]
        kept += len(note_result["sentences"])
        total_sentences += note_result.get("total", 0)
        removed_note += len(note_result["removed"])
        invalid += bool(note_result.get("error"))

        for item in row["questions"]:
            result = note.ask_documents(row["case_id"], item["question"], settings, save=False)
            known_total += 1
            known_right += _normal(item["answer"]) in _normal(result["answer"])
            removed_ask += len(result["removed"])
        result = note.ask_documents(row["case_id"], row["unanswerable_question"], settings, save=False)
        unknown_right += result["answer"] == NOT_FOUND
        removed_ask += len(result["removed"])
        print(f"  evaluated {row['case_id']} ({number}/{len(rows)}), {time.time() - started:.0f} s so far, {llm.STATS}", flush=True)

    per_type = {kind: _prf(*counts) for kind, counts in sorted(mismatch_counts.items())}
    totals = [sum(counts[i] for counts in mismatch_counts.values()) for i in range(3)]
    _save(settings, "checks", {
        "data": "sample", "cases": len(rows), "limited": limit is not None,
        "mismatch_rate_fraud": settings.mismatch_rate_fraud, "mismatch_rate_genuine": settings.mismatch_rate_genuine,
        "mismatches_per_type": per_type, "mismatches_all": _prf(*totals),
        "coverage_accuracy": _share(coverage_right, len(rows)), "coverage_confusion": confusion,
        "red_flags": _prf(flag_tp, flag_fp, flag_fn),
        "fact_field_accuracy": _share(fact_right, fact_total), "fact_fields": fact_total,
    })
    _save(settings, "answers", {
        "data": "sample", "cases": len(rows), "limited": limit is not None,
        "known_answers_correct": _share(known_right, known_total), "known_questions": known_total,
        "not_found_correct": _share(unknown_right, len(rows)), "unanswerable_questions": len(rows),
        "note_sentences_kept_share": _share(kept, total_sentences), "note_sentences": total_sentences,
        "note_sentences_removed": removed_note, "answers_removed": removed_ask, "notes_failed": invalid,
        "model": settings.llm_model, "real_model_calls": llm.STATS["real_calls"],
        "cached_model_calls": llm.STATS["cached_calls"], "model_seconds": round(llm.STATS["seconds"], 1),
    })


# ----- RESULTS.md -----


def _value(value: Any) -> str:
    return "not available" if value is None else str(value)


def _table(rows: list[tuple[str, Any, Any, str]]) -> list[str]:
    lines = ["| Metric | Value | n | Data |", "|---|---|---|---|"]
    lines += [f"| {name} | {_value(value)} | {_value(n)} | {data} |" for name, value, n, data in rows]
    return lines


def _risk_section(risk: dict[str, Any]) -> list[str]:
    lines = ["## Risk model", "", "5-fold cross-validation on the rows set aside for the risk model. Sex and MaritalStatus are not used.", ""]
    rows: list[tuple[str, Any, Any, str]] = []
    for type_name, result in sorted(risk.items()):
        for model_name, metrics in sorted(result["metrics"].items()):
            label = f"{type_name}, {model_name.replace('_', ' ')}"
            rows.append((f"{label}: ROC-AUC", metrics["roc_auc"], result["rows"], result["data"]))
            rows.append((f"{label}: PR-AUC", metrics["pr_auc"], result["rows"], result["data"]))
            rows.append((f"{label}: recall among the top tenth riskiest", metrics["recall_top_10pct"], result["rows"], result["data"]))
    return lines + _table(rows) + [""]


def _retrieval_section(retrieval: dict[str, Any]) -> list[str]:
    rows: list[tuple[str, Any, Any, str]] = []
    data = retrieval["data"]
    for label, key in (("with reranker", "with_reranker"), ("without reranker", "without_reranker")):
        part = retrieval[key]
        rows += [
            (f"Coverage clause found, {label}", part["coverage_clause_hit_rate"], part["coverage_clauses"], data),
            (f"Expected warning sign found, {label}", part["red_flag_hit_rate"], part["red_flags"], data),
            (f"Look-alike past case found, {label}", part["lookalike_hit_rate"], part["lookalikes"], data),
            (f"Passage with known answer found, {label}", part["known_answer_hit_rate"], part["known_answers"], data),
            (f"Answerable questions passing min_score, {label}", part["answerable_passing_min_score"], part["known_answers"], data),
            (f"Unanswerable questions stopped by min_score, {label}", part["unanswerable_stopped_by_min_score"], part["cases"], data),
        ]
    intro = ("Each search keeps the top passages after reranking, as set in config/settings.yaml "
             "(past cases keep a longer list). Warning signs are searched once per mismatch and per statement sentence.")
    head = ["## Retrieval", ""]
    if retrieval.get("limited"):
        head += ["This run covered only part of the open cases.", ""]
    return head + [intro, ""] + _table(rows) + [""]


def _checks_section(results: dict[str, Any]) -> list[str]:
    data, cases = results["data"], results["cases"]
    rows: list[tuple[str, Any, Any, str]] = []
    for kind, metric in sorted(results["mismatches_per_type"].items()):
        rows.append((f"Mismatch {kind}: precision", metric["precision"], metric["planted"], data))
        rows.append((f"Mismatch {kind}: recall", metric["recall"], metric["planted"], data))
    overall = results["mismatches_all"]
    rows += [
        ("All mismatches: precision", overall["precision"], overall["planted"], data),
        ("All mismatches: recall", overall["recall"], overall["planted"], data),
        ("Coverage status accuracy", results["coverage_accuracy"], cases, data),
        ("Warning signs: precision", results["red_flags"]["precision"], results["red_flags"]["planted"], data),
        ("Warning signs: recall", results["red_flags"]["recall"], results["red_flags"]["planted"], data),
        ("Fact fields read correctly", results["fact_field_accuracy"], results["fact_fields"], data),
    ]
    head = ["## Checks", ""]
    if results.get("limited"):
        head += ["This run covered only part of the open cases.", ""]
    if "mismatch_rate_genuine" in results:
        head += [f"Mismatches were planted in the sample cases at a chosen rate: {results['mismatch_rate_fraud']} of cases "
                 f"with a fraud label and {results['mismatch_rate_genuine']} of the others. These rates do not describe real cases.", ""]
    head += ["For mismatches and warning signs, n is the number planted, so a small n means a rough estimate.", ""]
    return head + _table(rows) + [""]


def _answers_section(results: dict[str, Any]) -> list[str]:
    data = results["data"]
    rows: list[tuple[str, Any, Any, str]] = [
        ("Known questions answered correctly", results["known_answers_correct"], results["known_questions"], data),
        ("Unanswerable questions answered \"" + NOT_FOUND.rstrip(".") + "\"", results["not_found_correct"], results["unanswerable_questions"], data),
        ("Note sentences kept by the citation check", results["note_sentences_kept_share"], results["note_sentences"], data),
        ("Note sentences removed", results["note_sentences_removed"], results["note_sentences"], data),
        ("Answers removed by the citation check", results["answers_removed"], results["known_questions"], data),
    ]
    tail = [f"Language model: {results['model']}. Real calls in this run: {results['real_model_calls']}; "
            f"reused from the cache: {results['cached_model_calls']}; model time in seconds: {results['model_seconds']}.", ""]
    return ["## Answers and notes", ""] + _table(rows) + [""] + tail


def write_results(settings: Settings) -> str:
    """Write results/RESULTS.md from the JSON files only."""
    info = _data_note(settings)
    lines = [
        "# Tallyline results", "",
        "Case documents, past cases and knowledge texts are synthetic sample data written for this project. "
        "The structured fields and fraud labels of the risk model come from the real dataset.", "",
    ]
    no_dataset = sorted(name for name, real in (info or {}).get("real_dataset", {}).items() if not real)
    if no_dataset:
        lines += [f"These insurance types have no dataset, so their dates and fraud labels are generated and they have "
                  f"no risk model: {', '.join(no_dataset)}.", ""]
    if info:
        lines += [f"Free text in the case documents was written by: {info.get('writer')}.", ""]
    risk = _load_json(settings, "risk_model")
    lines += _risk_section(risk) if risk else ["## Risk model", "", "Not run yet.", ""]
    retrieval = _load_json(settings, "retrieval")
    lines += _retrieval_section(retrieval) if retrieval else ["## Retrieval", "", "Not run yet.", ""]
    results = _load_json(settings, "checks")
    lines += _checks_section(results) if results else ["## Checks", "", "Not run yet: needs the language model.", ""]
    answers = _load_json(settings, "answers")
    lines += _answers_section(answers) if answers else ["## Answers and notes", "", "Not run yet: needs the language model.", ""]
    path = settings.results_dir / "RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def run_all(settings: Settings, limit: int | None = None, skip_llm: bool = False) -> None:
    """Run every evaluation and write RESULTS.md."""
    if not (settings.results_dir / "risk_model.json").exists():
        print(risk_model.train_all(settings))
    if not rag.indexes_built(settings):
        print(rag.build_indexes(settings))
    print("Evaluating retrieval (no language model)...", flush=True)
    evaluate_retrieval(settings, limit)
    if not skip_llm:
        status = llm.check_ollama(settings)
        if status["reachable"] and status["model_present"]:
            print(f"Evaluating checks and answers with {settings.llm_model}...", flush=True)
            evaluate_llm(settings, limit)
        else:
            print(f"Language model {settings.llm_model} is not available; skipping checks and answers.")
    print(f"Wrote {write_results(settings)}")
    print(f"Language model use in this run: {llm.STATS}")
