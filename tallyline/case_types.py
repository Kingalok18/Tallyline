"""Load and validate case-type files from config/case_types/."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .settings import Settings

REQUIRED_KEYS = ("name", "label", "documents", "terms", "fact_fields", "knowledge_folder")
REQUIRED_DOCUMENTS = ("statement", "assessment_report", "invoice")
OPTIONAL_DOCUMENTS = ("incident_report",)
REQUIRED_TERMS = ("provider", "items", "loss_detail")
GENERAL_FACT_FIELDS = (
    "incident_date", "location", "incident_description", "loss_detail", "items_assessed",
    "items_billed", "estimate", "invoice_total", "invoice_date", "provider", "incident_report_filed",
)


class CaseTypeError(ValueError):
    """Raised when a case-type file is missing something the checks rely on."""


@dataclass
class CaseType:
    """One kind of insurance, described entirely by its YAML file."""

    name: str
    label: str
    documents: dict[str, str]
    terms: dict[str, str]
    fact_fields: list[str]
    knowledge_folder: str
    risk_model: dict[str, Any] | None = None
    sample_content: str | None = None
    clause_prefixes: dict[str, str] | None = None  # how clause IDs show their kind: cover, limit, exclusion

    def term(self, key: str) -> str:
        """Return the label this case type uses for a general word."""
        return self.terms[key]


def parse_case_type(data: dict[str, Any], source: str = "case type") -> CaseType:
    """Validate a parsed YAML mapping and build a CaseType."""
    missing = [k for k in REQUIRED_KEYS if not data.get(k)]
    if missing:
        raise CaseTypeError(f"{source}: missing keys {missing}")
    documents = data["documents"]
    bad_docs = [d for d in REQUIRED_DOCUMENTS if d not in documents]
    if bad_docs:
        raise CaseTypeError(f"{source}: missing documents {bad_docs}")
    unknown_docs = set(documents) - set(REQUIRED_DOCUMENTS) - set(OPTIONAL_DOCUMENTS)
    if unknown_docs:
        raise CaseTypeError(f"{source}: unknown document keys {sorted(unknown_docs)}")
    bad_terms = [t for t in REQUIRED_TERMS if t not in data["terms"]]
    if bad_terms:
        raise CaseTypeError(f"{source}: missing terms {bad_terms}")
    bad_facts = [f for f in GENERAL_FACT_FIELDS if f not in data["fact_fields"]]
    if bad_facts:
        raise CaseTypeError(f"{source}: fact_fields is missing {bad_facts}")
    risk = data.get("risk_model")
    if risk is not None and not all(risk.get(k) for k in ("dataset", "label_column")):
        raise CaseTypeError(f"{source}: risk_model needs dataset and label_column")
    return CaseType(
        name=data["name"], label=data["label"], documents=dict(documents), terms=dict(data["terms"]),
        fact_fields=list(data["fact_fields"]), knowledge_folder=data["knowledge_folder"],
        risk_model=risk, sample_content=data.get("sample_content"), clause_prefixes=data.get("clause_prefixes"),
    )


def load_case_type(name: str, settings: Settings) -> CaseType:
    """Load config/case_types/<name>.yaml."""
    path: Path = settings.case_types_dir / f"{name}.yaml"
    if not path.exists():
        raise CaseTypeError(f"No case-type file at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    case_type = parse_case_type(data, source=path.name)
    if case_type.name != name:
        raise CaseTypeError(f"{path.name}: name must be '{name}', found '{case_type.name}'")
    return case_type


def load_case_types(settings: Settings) -> dict[str, CaseType]:
    """Load every case type listed in settings."""
    return {name: load_case_type(name, settings) for name in settings.case_types}
