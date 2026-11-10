"""Tallyline commands: setup | data | index | start | eval.

    python run.py setup    create .venv and install requirements
    python run.py data     load the dataset, train the risk model, build sample data
    python run.py index    build the three indexes
    python run.py start    start the web app and open the browser
    python run.py eval     all metrics -> results/RESULTS.md
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["ANONYMIZED_TELEMETRY"] = "False"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _ensure_venv_python() -> None:
    """Re-run this script with the project's virtual environment."""
    if Path(sys.prefix).resolve() == VENV.resolve():
        os.environ.update(_env())
        sys.path.insert(0, str(ROOT))
        return
    if not VENV_PY.exists():
        sys.exit("The virtual environment is missing. Run: python run.py setup")
    sys.exit(subprocess.call([str(VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]], env=_env()))


def cmd_setup(_: argparse.Namespace) -> None:
    if not VENV_PY.exists():
        venv.EnvBuilder(with_pip=True).create(VENV)
    subprocess.check_call([str(VENV_PY), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
    print("Setup finished. Next: python run.py start")


def cmd_data(args: argparse.Namespace) -> None:
    _ensure_venv_python()
    from tallyline import db, risk_model, sample_data
    from tallyline.settings import load_settings

    settings = load_settings()
    summary = sample_data.build_sample_data(settings, writer=args.writer)
    print(summary)
    print(risk_model.train_all(settings))
    print(f"Cases loaded: {db.sync_cases(settings)}; past cases loaded: {db.sync_past_cases(settings)}")


def cmd_index(args: argparse.Namespace) -> None:
    _ensure_venv_python()
    from tallyline import rag
    from tallyline.settings import load_settings

    print(rag.build_indexes(load_settings(), force=args.force))


def cmd_start(args: argparse.Namespace) -> None:
    _ensure_venv_python()
    import threading
    import webbrowser

    import uvicorn

    from tallyline import db, rag
    from tallyline.settings import load_settings

    settings = load_settings()
    if not any(settings.cases_dir.glob("*/case.json")):
        print("First start: building sample data. This can take a while on a CPU.")
        cmd_data(argparse.Namespace(writer=None))
    db.sync_cases(settings)
    db.sync_past_cases(settings)
    if not rag.indexes_built(settings):
        print("First start: building the indexes.")
        print(rag.build_indexes(settings))
    url = f"http://127.0.0.1:{args.port}"
    print(f"Tallyline is running at {url}  (press Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, [url]).start()
    uvicorn.run("tallyline.web:app", host="127.0.0.1", port=args.port, log_level="warning")


def cmd_eval(args: argparse.Namespace) -> None:
    _ensure_venv_python()
    from tallyline import evaluate
    from tallyline.settings import load_settings

    evaluate.run_all(load_settings(), limit=args.limit, skip_llm=args.skip_llm)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup").set_defaults(func=cmd_setup)
    data = sub.add_parser("data")
    data.add_argument("--writer", choices=["llm", "template"], default=None, help="override document_writer")
    data.set_defaults(func=cmd_data)
    index = sub.add_parser("index")
    index.add_argument("--force", action="store_true", help="rebuild everything")
    index.set_defaults(func=cmd_index)
    start = sub.add_parser("start")
    start.add_argument("--port", type=int, default=8000)
    start.add_argument("--no-browser", action="store_true")
    start.set_defaults(func=cmd_start)
    evaluation = sub.add_parser("eval")
    evaluation.add_argument("--limit", type=int, default=None, help="only evaluate the first N open cases")
    evaluation.add_argument("--skip-llm", action="store_true", help="only the metrics that need no language model")
    evaluation.set_defaults(func=cmd_eval)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
