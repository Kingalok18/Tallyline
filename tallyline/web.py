"""The web app: FastAPI routes, Jinja2 templates and HTMX partial updates."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import NOT_FOUND, case_intake, checks, db, llm, note
from .case_types import load_case_types
from .render import render_markdown
from .settings import PROJECT_ROOT, Settings, load_settings

DECISIONS = {
    "pay": "Pay", "ask_for_documents": "Ask for documents", "investigate": "Investigate", "reject": "Reject",
}
MAX_UPLOAD_BYTES = 30 * 1024 * 1024
COVERAGE_LABELS = {"covered": "Covered", "excluded": "Excluded", "partly": "Partly covered", "unclear": "Unclear"}
OUTCOME_LABELS = {
    "paid": "Paid", "paid_reduced": "Paid at a reduced amount", "rejected": "Rejected",
    "sent_for_investigation": "Sent for investigation",
}


class Citations:
    """Gives each cited passage a stable small number within one page section."""

    def __init__(self) -> None:
        self.numbers: dict[str, int] = {}

    def num(self, passage_id: str) -> int:
        return self.numbers.setdefault(passage_id, len(self.numbers) + 1)


def _cited_ids(findings: dict[str, Any]) -> list[str]:
    """Every case-document passage cited anywhere in the findings."""
    ids = list(findings.get("coverage", {}).get("passages", []))
    for mismatch in findings.get("mismatches", []):
        ids += mismatch["passages"]
    for flag in findings.get("red_flags", []):
        ids += [flag["guide_passage"], flag["case_passage"]]
    for sentence in findings.get("note", {}).get("sentences", []):
        ids += sentence["passages"]
    return [i for i in ids if i]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app for the given settings."""
    settings = settings or load_settings(os.environ.get("TALLYLINE_ROOT"))
    app = FastAPI(title="Tallyline", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"), name="static")
    templates = Jinja2Templates(directory=PROJECT_ROOT / "templates")
    jobs: dict[str, dict[str, Any]] = {}
    jobs_lock = threading.Lock()
    status_cache: dict[str, Any] = {"at": 0.0, "value": None}

    def ollama_status() -> dict[str, Any]:
        if time.time() - status_cache["at"] > 10:
            status_cache.update(at=time.time(), value=llm.check_ollama(settings))
        return status_cache["value"]

    def base_context(request: Request) -> dict[str, Any]:
        types = load_case_types(settings)
        sample = db.query(settings, "SELECT 1 FROM cases WHERE is_sample = 1 LIMIT 1")
        return {
            "request": request, "status": ollama_status(), "settings": settings, "case_types": types,
            "sample_data": bool(sample), "COVERAGE_LABELS": COVERAGE_LABELS, "OUTCOME_LABELS": OUTCOME_LABELS,
            "DECISIONS": DECISIONS,
        }

    def render(request: Request, name: str, **context: Any) -> HTMLResponse:
        return templates.TemplateResponse(request, name, {**base_context(request), **context})

    def get_case(case_id: str) -> dict[str, Any]:
        try:
            return checks.load_case(case_id, settings)
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown case") from None

    def get_findings(case_id: str) -> dict[str, Any] | None:
        rows = db.query(settings, "SELECT result_json FROM analyses WHERE case_id = ?", (case_id,))
        return json.loads(rows[0]["result_json"]) if rows else None

    def document_html(case_id: str, doc: str, findings: dict[str, Any] | None) -> str:
        path = settings.cases_dir / case_id / f"{doc}.md"
        ids = _cited_ids(findings) if findings else []
        prefix = f"case_docs:{case_id}:{doc}:"
        marks = [i for i in ids if i.startswith(prefix)]
        texts = []
        if marks:
            holders = ",".join("?" * len(marks))
            texts = [r["text"] for r in db.query(settings, f"SELECT text FROM passages WHERE id IN ({holders})", marks)]
        return render_markdown(path.read_text(encoding="utf-8"), texts)

    # ----- case list -----

    def list_cases(q: str, case_type: str, risk: str) -> list[dict[str, Any]]:
        sql = "SELECT * FROM cases WHERE 1=1"
        params: list[Any] = []
        if q:
            sql += " AND (id LIKE ? OR provider LIKE ? OR main_issue LIKE ? OR status LIKE ?)"
            params += [f"%{q}%"] * 4
        if case_type:
            sql += " AND case_type = ?"
            params.append(case_type)
        if risk == "Not analysed":
            sql += " AND risk_level IS NULL"
        elif risk:
            sql += " AND risk_level = ?"
            params.append(risk)
        return db.query(settings, sql + " ORDER BY id", params)

    @app.get("/", response_class=HTMLResponse)
    def case_list(request: Request, q: str = "", case_type: str = "", risk: str = "") -> HTMLResponse:
        rows = list_cases(q, case_type, risk)
        if request.headers.get("HX-Request"):
            return render(request, "_case_rows.html", cases=rows)
        return render(request, "case_list.html", cases=rows, q=q, selected_type=case_type, selected_risk=risk)

    # ----- adding a case (before /cases/{case_id}, which would otherwise match "new") -----

    @app.get("/cases/new", response_class=HTMLResponse)
    def new_case_form(request: Request) -> HTMLResponse:
        rows = db.query(settings, "SELECT currency FROM cases WHERE currency != '' GROUP BY currency ORDER BY COUNT(*) DESC LIMIT 1")
        return render(request, "case_new.html", values={"currency": rows[0]["currency"] if rows else ""}, error=None)

    @app.post("/cases/new", response_class=HTMLResponse)
    def new_case(
        request: Request,
        case_type: str = Form(""), case_id: str = Form(""), policy_number: str = Form(""), policy_start: str = Form(""),
        incident_date: str = Form(""), amount: str = Form(""), currency: str = Form(""), provider: str = Form(""),
        statement: UploadFile | None = File(None), assessment_report: UploadFile | None = File(None),
        invoice: UploadFile | None = File(None), incident_report: UploadFile | None = File(None),
    ) -> Any:
        form = {
            "case_type": case_type, "case_id": case_id, "policy_number": policy_number, "policy_start": policy_start,
            "incident_date": incident_date, "amount": amount, "currency": currency, "provider": provider,
        }
        uploads = {
            "statement": statement, "assessment_report": assessment_report,
            "invoice": invoice, "incident_report": incident_report,
        }
        files = {
            doc: (upload.filename, upload.file.read(case_intake.MAX_UPLOAD_BYTES + 1))
            for doc, upload in uploads.items() if upload is not None and upload.filename
        }
        try:
            new_id = case_intake.add_case(form, files, settings)
        except case_intake.CaseFormError as exc:
            page = render(request, "case_new.html", values=form, error=str(exc))
            page.status_code = 422
            return page
        return RedirectResponse(f"/cases/{new_id}", status_code=303)

    # ----- case page -----

    @app.get("/cases/{case_id}", response_class=HTMLResponse)
    def case_page(request: Request, case_id: str) -> HTMLResponse:
        case = get_case(case_id)
        findings = get_findings(case_id)
        asks = db.query(settings, "SELECT answer_json FROM asks WHERE case_id = ? ORDER BY id DESC", (case_id,))
        decisions = db.query(settings, "SELECT * FROM decisions WHERE case_id = ? ORDER BY id DESC LIMIT 1", (case_id,))
        first_doc = next(iter(case["documents"]), None)
        return render(
            request, "case.html", case=case["record"], case_type=case["case_type"], documents=case["documents"],
            first_doc=first_doc, doc_html=document_html(case_id, first_doc, findings) if first_doc else "",
            findings=findings, cite=Citations(), job=jobs.get(case_id) if jobs.get(case_id, {}).get("state") == "running" else None,
            answers=[json.loads(a["answer_json"]) for a in asks], Citations=Citations,
            decision=decisions[0] if decisions else None,
        )

    @app.get("/cases/{case_id}/documents/{doc}", response_class=HTMLResponse)
    def case_document(request: Request, case_id: str, doc: str) -> HTMLResponse:
        case = get_case(case_id)
        if doc not in case["documents"]:
            raise HTTPException(status_code=404, detail="Unknown document")
        return HTMLResponse(document_html(case_id, doc, get_findings(case_id)))

    def run_job(case_id: str) -> None:
        def progress(step: str) -> None:
            jobs[case_id]["step"] = step

        try:
            note.full_analysis(case_id, settings, progress)
            jobs[case_id].update(state="done")
        except Exception as exc:  # shown to the user in the findings area
            jobs[case_id].update(state="failed", error=str(exc))

    @app.post("/cases/{case_id}/analyse", response_class=HTMLResponse)
    def analyse(request: Request, case_id: str) -> HTMLResponse:
        get_case(case_id)
        with jobs_lock:
            if jobs.get(case_id, {}).get("state") != "running":
                jobs[case_id] = {"state": "running", "step": "Starting"}
                threading.Thread(target=run_job, args=(case_id,), daemon=True).start()
        return render(request, "_progress.html", case_id=case_id, job=jobs[case_id])

    @app.get("/cases/{case_id}/analysis-status", response_class=HTMLResponse)
    def analysis_status(request: Request, case_id: str) -> HTMLResponse:
        case = get_case(case_id)
        job = jobs.get(case_id)
        if job and job["state"] == "running":
            return render(request, "_progress.html", case_id=case_id, job=job)
        error = job.get("error") if job else None
        findings = get_findings(case_id)
        return render(
            request, "_findings.html", case=case["record"], case_type=case["case_type"], findings=findings,
            cite=Citations(), error=error, reload_doc=next(iter(case["documents"]), None),
        )

    @app.post("/cases/{case_id}/ask", response_class=HTMLResponse)
    def ask(request: Request, case_id: str, question: str = Form(...), include_other: str = Form("")) -> HTMLResponse:
        get_case(case_id)
        question = question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="Question is empty")
        answer = note.ask_documents(case_id, question, settings, include_knowledge=bool(include_other))
        return render(request, "_answer_card.html", answer=answer, cite=Citations())

    @app.post("/cases/{case_id}/decision", response_class=HTMLResponse)
    def decide(request: Request, case_id: str, decision: str = Form(...), comment: str = Form("")) -> HTMLResponse:
        get_case(case_id)
        if decision not in DECISIONS:
            raise HTTPException(status_code=422, detail="Unknown decision")
        db.execute(settings, "INSERT INTO decisions (case_id, decision, comment, created_at) VALUES (?, ?, ?, ?)",
                   (case_id, decision, comment.strip(), db.now()))
        db.execute(settings, "UPDATE cases SET status = ? WHERE id = ?", (DECISIONS[decision], case_id))
        latest = db.query(settings, "SELECT * FROM decisions WHERE case_id = ? ORDER BY id DESC LIMIT 1", (case_id,))[0]
        return render(request, "_decision.html", decision=latest)

    # ----- passages -----

    def _passage_body(passage: dict[str, Any]) -> tuple[str, str]:
        """The passage's document as HTML with the passage highlighted, plus a title."""
        metadata = passage["metadata"]
        if passage["index_name"] == "case_docs":
            path = settings.cases_dir / metadata["case_id"] / f"{metadata['doc_type']}.md"
            title = metadata["doc_type"].replace("_", " ").capitalize()
            types = load_case_types(settings)
            if metadata.get("case_type") in types:
                title = types[metadata["case_type"]].documents.get(metadata["doc_type"], title)
            return render_markdown(path.read_text(encoding="utf-8"), [passage["text"]]), title
        if passage["index_name"] == "knowledge":
            from .rag import read_text_file

            path = Path(metadata.get("path", ""))
            if path.exists() and settings.knowledge_dir.resolve() in path.resolve().parents:
                return render_markdown(read_text_file(path), [passage["text"]]), metadata.get("file", "Guide")
            return render_markdown(passage["text"]), metadata.get("file", "Guide")
        return render_markdown(passage["text"], [passage["text"]]), f"Past case {metadata.get('past_case_id', '')}"

    @app.get("/passages/{passage_id:path}", response_class=HTMLResponse)
    def passage_view(request: Request, passage_id: str) -> HTMLResponse:
        rows = db.query(settings, "SELECT * FROM passages WHERE id = ?", (passage_id,))
        if not rows:
            raise HTTPException(status_code=404, detail="Unknown passage")
        passage = rows[0]
        passage["metadata"] = json.loads(passage.pop("metadata_json"))
        body, title = _passage_body(passage)
        return render(request, "_passage_panel.html", passage=passage, body=body, title=title)

    # ----- past cases -----

    @app.get("/past-cases/{past_id}", response_class=HTMLResponse)
    def past_case(request: Request, past_id: str, case: str = "") -> HTMLResponse:
        rows = db.query(settings, "SELECT * FROM past_cases WHERE id = ?", (past_id,))
        if not rows:
            raise HTTPException(status_code=404, detail="Unknown past case")
        record = rows[0]
        current = get_case(case) if case else None
        types = load_case_types(settings)
        facts = (get_findings(case) or {}).get("facts", {}) if case else {}
        return render(
            request, "past_case.html", past=record, current=current["record"] if current else None,
            current_facts=facts.get("statement", {}), case_type=types.get(record["case_type"]),
        )

    # ----- knowledge -----

    def knowledge_groups() -> list[dict[str, Any]]:
        from .rag import _knowledge_files

        counts = {
            row["key"]: row["n"]
            for row in db.query(
                settings,
                """SELECT json_extract(metadata_json, '$.source_key') AS key, COUNT(*) AS n
                   FROM passages GROUP BY key""",
            )
        }
        groups: dict[str, dict[str, Any]] = {}
        types = load_case_types(settings)
        for name, case_type in types.items():
            groups[name] = {"name": name, "label": case_type.label, "files": []}
        groups["general"] = {"name": "general", "label": "General (all kinds of insurance)", "files": []}
        for path, type_name, source in _knowledge_files(settings):
            key = f"knowledge:{type_name}:{source}:{path.name}"
            groups[type_name]["files"].append({"name": path.name, "source": source, "passages": counts.get(key, 0)})
        return list(groups.values())

    def index_counts() -> dict[str, int]:
        rows = db.query(settings, "SELECT index_name, COUNT(*) AS n FROM passages GROUP BY index_name")
        return {r["index_name"]: r["n"] for r in rows}

    @app.get("/knowledge", response_class=HTMLResponse)
    def knowledge(request: Request) -> HTMLResponse:
        return render(request, "knowledge.html", groups=knowledge_groups(), counts=index_counts())

    @app.post("/knowledge/upload", response_class=HTMLResponse)
    def knowledge_upload(
        request: Request, file: UploadFile = File(...), case_type: str = Form(...), source: str = Form(...)
    ) -> HTMLResponse:
        types = load_case_types(settings)
        name = re.sub(r"[^A-Za-z0-9._ -]+", "_", Path(file.filename or "").name).strip()
        valid_target = (case_type == "general" and source == "red_flags") or (
            case_type in types and source in {"policy", "red_flags"}
        )
        if not valid_target:
            message = "Choose a valid insurance type and folder. General documents can only be warning-sign guides."
        elif not name.lower().endswith(".pdf"):
            message = "Only PDF files can be added here."
        else:
            data = file.file.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES or not data.startswith(b"%PDF"):
                message = "That file is not a PDF of 30 MB or less."
            else:
                base = settings.knowledge_dir / "general" if case_type == "general" else settings.root / types[case_type].knowledge_folder
                folder = base / source
                folder.mkdir(parents=True, exist_ok=True)
                (folder / name).write_bytes(data)
                message = f"Saved {name}. Rebuild the index to include it."
        return render(request, "_knowledge_message.html", message=message)

    @app.post("/knowledge/rebuild", response_class=HTMLResponse)
    def knowledge_rebuild(request: Request) -> HTMLResponse:
        from . import rag

        return render(request, "_knowledge_message.html", message=rag.build_indexes(settings))

    @app.get("/knowledge/search", response_class=HTMLResponse)
    def knowledge_search(request: Request, q: str = "", index: str = "knowledge") -> HTMLResponse:
        from . import rag

        if index not in rag.INDEXES:
            raise HTTPException(status_code=422, detail="Unknown index")
        results = rag.retrieve(index, q, settings=settings, min_score=rag.NO_MIN_SCORE) if q.strip() else []
        return render(request, "_search_results.html", results=results, q=q)

    # ----- health -----
    # There is no results page: the user chose to keep the measured numbers in
    # results/RESULTS.md only, outside the app.

    @app.get("/health")
    def health() -> JSONResponse:
        from .rag import indexes_built

        status = llm.check_ollama(settings)
        return JSONResponse({
            "ollama_reachable": status["reachable"], "model": status["model"],
            "model_present": status["model_present"], "index_built": indexes_built(settings),
        })

    return app


app = create_app()
__all__ = ["app", "create_app", "NOT_FOUND"]
