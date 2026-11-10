"""Load config/settings.yaml into a small typed object with derived paths."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Never used by any model, whatever a case-type file says.
PROTECTED_COLUMNS = ("Sex", "MaritalStatus")


@dataclass
class Settings:
    """All tunable values. config/settings.yaml overrides these defaults."""

    root: Path = PROJECT_ROOT
    llm_model: str = "qwen2.5:3b-instruct"
    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_reranker: bool = True
    chunk_size: int = 300
    chunk_overlap: int = 30
    top_k: int = 10
    top_n_after_rerank: int = 4
    min_score: float = 0.2
    rerank_temperature: float = 5.0
    red_flag_min_score: float = 0.4
    seed: int = 42
    open_cases: int = 50
    past_cases: int = 200
    mismatch_rate_fraud: float = 0.6
    mismatch_rate_genuine: float = 0.3
    case_types: list[str] = field(default_factory=lambda: [])
    case_type_shares: dict[str, float] = field(default_factory=dict)  # how sample cases split across types
    theme: str = "system"
    document_writer: str = "llm"
    lookalike_share: float = 0.15
    coverage_rate_excluded: float = 0.15
    coverage_rate_partly: float = 0.10
    early_incident_days: int = 30
    invoice_over_estimate_ratio: float = 1.3
    provider_repeat_threshold: int = 5
    provider_recent_days: int = 90
    risk_weights: dict[str, float] = field(
        default_factory=lambda: {"model": 0.35, "mismatches": 0.30, "red_flags": 0.20, "past_cases": 0.15}
    )
    risk_mismatches_full: int = 3
    risk_red_flags_full: int = 5
    risk_level_medium: int = 34
    risk_level_high: int = 67
    llm_num_ctx: int = 4096
    llm_timeout_seconds: int = 600

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def case_types_dir(self) -> Path:
        return self.config_dir / "case_types"

    @property
    def prompts_dir(self) -> Path:
        return self.root / "prompts"

    @property
    def knowledge_dir(self) -> Path:
        return self.root / "knowledge"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def cases_dir(self) -> Path:
        return self.data_dir / "cases"

    @property
    def past_cases_path(self) -> Path:
        return self.data_dir / "past_cases.jsonl"

    @property
    def answer_key_path(self) -> Path:
        return self.data_dir / "answer_key.jsonl"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "tallyline.db"

    @property
    def llm_cache_path(self) -> Path:
        return self.data_dir / "llm_cache.db"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    def prompt(self, name: str) -> str:
        """Return the text of prompts/<name>.txt."""
        return (self.prompts_dir / f"{name}.txt").read_text(encoding="utf-8")


def load_settings(root: Path | str | None = None, **overrides: Any) -> Settings:
    """Read settings.yaml under `root` (default: the project) and apply overrides."""
    root_path = Path(root) if root else PROJECT_ROOT
    path = root_path / "config" / "settings.yaml"
    values: dict[str, Any] = {}
    if path.exists():
        values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    known = {f.name for f in fields(Settings)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
    values.update(overrides)
    return Settings(root=root_path, **values)
