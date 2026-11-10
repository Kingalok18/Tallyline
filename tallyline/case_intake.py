"""Add a case from uploaded documents, for the Add case page."""
from __future__ import annotations

import json
import re
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any

from . import checks, db, rag
from .case_types import load_case_types
from .sample_data import CASE_PREFIX
from .settings import Settings

MAX_UPLOAD_BYTES = 30 * 1024 * 1024
MAX_DOCUMENT_CHARACTERS = 100_000
MIN_DOCUMENT_WORDS = 5
ADDED_CASE_PREFIX = "NC-"
_CASE_ID = re.compile(r"[A-Z]{1,8}-\d{1,8}")  # also rules out Windows device names such as CON


class CaseFormError(ValueError):
    """Something in the form cannot be accepted. The message is shown to the user."""


def document_text(filename: str, data: bytes) -> str:
    """The text of an uploaded PDF, Markdown or text file."""
    name = Path(filename).name
    suffix = Path(name).suffix.lower()
    if len(data) > MAX_UPLOAD_BYTES:
        raise CaseFormError(f"{name} is larger than 30 MB.")
    if suffix == ".pdf":
        if not data.startswith(b"%PDF"):
            raise CaseFormError(f"{name} is not a valid PDF.")
        from pypdf import PdfReader

        try:
            text = "\n\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(data)).pages)
        except Exception as exc:  # pypdf raises many error types on damaged or locked files
            raise CaseFormError(f"{name} could not be read as a PDF.") from exc
    elif suffix in {".md", ".txt"}:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise CaseFormError(f"{name} is not a UTF-8 text file.") from exc
    else:
        raise CaseFormError(f"{name}: only PDF, Markdown (.md) and text (.txt) files can be added.")
    text = text.strip()
    if len(text.split()) < MIN_DOCUMENT_WORDS:
        raise CaseFormError(f"{name} has no readable text. A scanned PDF holds only images, so it cannot be read.")
    if len(text) > MAX_DOCUMENT_CHARACTERS:
        raise CaseFormError(f"{name} is too long. The limit is {MAX_DOCUMENT_CHARACTERS:,} characters.")
    return text


def next_case_id(settings: Settings) -> str:
    """The next free ID for an added case."""
    used = {row["id"] for row in db.query(settings, "SELECT id FROM cases")}
    used |= {path.name for path in settings.cases_dir.iterdir()} if settings.cases_dir.exists() else set()
    number = 1
    while f"{ADDED_CASE_PREFIX}{number:04d}" in used:
        number += 1
    return f"{ADDED_CASE_PREFIX}{number:04d}"


def _single_line(value: str, label: str) -> str:
    text = " ".join(str(value).split())
    if not text:
        raise CaseFormError(f"{label} is required.")
    if len(text) > 120:
        raise CaseFormError(f"{label} is too long.")
    return text


def _date(value: str, label: str) -> str:
    parsed = checks.parse_date(_single_line(value, label))
    if not parsed:
        raise CaseFormError(f"{label} is not a valid date.")
    return parsed.isoformat()


def _case_id(value: str, settings: Settings) -> str:
    case_id = value.strip().upper()
    if not case_id:
        return next_case_id(settings)
    if not _CASE_ID.fullmatch(case_id):
        raise CaseFormError("Case ID must be letters, a hyphen and digits, for example MY-0001.")
    if case_id.startswith(CASE_PREFIX):
        raise CaseFormError(f"Case IDs starting with {CASE_PREFIX} are kept for the sample cases.")
    if db.query(settings, "SELECT 1 FROM cases WHERE id = ?", (case_id,)) or (settings.cases_dir / case_id).exists():
        raise CaseFormError(f"Case {case_id} already exists.")
    return case_id


def _record(form: dict[str, str], case_id: str, settings: Settings) -> dict[str, Any]:
    case_type = form.get("case_type", "")
    if case_type not in load_case_types(settings):
        raise CaseFormError("Choose an insurance type.")
    try:
        amount = float(form.get("amount", "").replace(",", ""))
    except ValueError:
        raise CaseFormError("Amount must be a number.") from None
    if not 0 <= amount < 1e12:
        raise CaseFormError("Amount must be zero or more.")
    return {
        "case_id": case_id, "case_type": case_type,
        "policy_number": _single_line(form.get("policy_number", ""), "Policy number"),
        "policy_start": _date(form.get("policy_start", ""), "Policy start date"),
        "incident_date": _date(form.get("incident_date", ""), "Incident date"),
        "amount": amount, "currency": _single_line(form.get("currency", ""), "Currency").upper(),
        "provider": _single_line(form.get("provider", ""), "Service provider"),
        "features": {}, "sample": False,
    }


def add_case(form: dict[str, str], files: dict[str, tuple[str, bytes]], settings: Settings) -> str:
    """Check the form, save the case and add its documents to the search index. Returns the case ID.

    `files` maps a document name such as "statement" to (file name, content). A failure
    leaves nothing behind.
    """
    case_id = _case_id(form.get("case_id", ""), settings)
    record = _record(form, case_id, settings)
    documents = load_case_types(settings)[record["case_type"]].documents
    if "statement" not in files:
        raise CaseFormError("The statement is required, because the checks start from it.")
    unknown = sorted(set(files) - set(documents))
    if unknown:
        raise CaseFormError(f"This insurance type has no document called {unknown[0]}.")
    texts = {doc: document_text(name, data) for doc, (name, data) in files.items()}

    folder = settings.cases_dir / case_id
    folder.mkdir(parents=True)
    try:
        for doc, text in texts.items():
            header = [f"# {documents[doc]}", f"Case: {case_id}", f"Policy number: {record['policy_number']}",
                      f"Policy start date: {record['policy_start']}", text]
            (folder / f"{doc}.md").write_text("\n\n".join(header) + "\n", encoding="utf-8")
        (folder / "case.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        db.sync_cases(settings)
        rag.build_indexes(settings)
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        db.sync_cases(settings)
        raise
    return case_id
