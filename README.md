# Tallyline

Tallyline helps an insurance investigator review a case. For each case it answers four questions, each backed by passages retrieved from documents and shown with citations:

1. Is the loss covered? (searches the policy wording)
2. Which fraud warning signs does the case show? (searches a warning-sign guide)
3. Do the case's own documents agree with each other? (compares the statement, assessment report, invoice and incident report)
4. Which past cases look similar, and how did they end? (searches past cases)

It then gives a risk score with its reasons, writes a short investigator note in which every sentence has a citation, and offers **Ask the documents**: type a question and get a cited answer, or "Not found in the documents."

Everything runs on your computer. Nothing is sent to an outside service.

## Starting the app

You need Python 3.11 and [Ollama](https://ollama.com).

```
python run.py setup
ollama pull qwen2.5:3b-instruct
python run.py start
```

`start` opens http://127.0.0.1:8000. The repository already holds the built sample data, indexes and analyses in `data/`, so the 50 cases open ready. Without that folder, `start` builds the sample data and indexes first, which takes hours on a CPU.

| Command | What it does |
|---|---|
| `python run.py setup` | Creates `.venv` and installs the requirements |
| `python run.py data` | Trains the risk model and builds the sample data. `--writer template` skips the language model |
| `python run.py index` | Builds the three indexes; unchanged files are skipped |
| `python run.py start` | Starts the web app and opens the browser. `--port` and `--no-browser` are available |
| `python run.py eval` | Analyses every sample case, so each opens ready in the app, and writes the measured numbers to `results/RESULTS.md`. The app has no results page: open that file directly. `--limit N` evaluates only N cases; `--skip-llm` runs only the metrics that need no language model |

### On a computer with little memory

The language model needs about 2.2 GB of free memory, and the embedding model and reranker need about 1 GB more. Close your browser and other large programs before long runs. On a CPU, analysing one case takes a few minutes. Language model replies are cached in `data/llm_cache.db`, so a run that gets interrupted resumes where it stopped when you run the same command again.

## Keyboard shortcuts

`/` search, `j` and `k` move in the case list, `Enter` open, `Esc` close the side panel, `a` go to the question box.

## Adding a case

Press **Add case** on the Cases page, fill in the details and choose the documents. PDF, Markdown and text files are accepted, up to 30 MB each. Only the statement is required, but the checks compare the documents, so a missing one means fewer checks.

Saving reads the documents and adds them to the search index, then opens the case so you can ask questions or run the analysis.

- **Scanned PDFs** hold only images, so they cannot be read. The page says so rather than saving an empty case.
- **The risk score leaves out the risk model** for an added case, because it has no dataset fields. Its weight is shared among the other parts.
- **Added cases stay when sample data is rebuilt.** They are not part of the evaluation, because they have no answer key.
- **The insurance type must already exist.** Its policy wording is what coverage is checked against, so a sample policy gives sample answers.
- **Long documents:** fact reading uses the first 6,000 characters of each document, while questions can search all of it, up to 100,000 characters.

## Adding a new insurance type

Nothing in the code is tied to one kind of insurance. To add one:

1. Copy `config/case_types/_template.yaml` to `config/case_types/<name>.yaml`. Set `name` to the same `<name>`, then fill in the labels and terms (for health: assessment report = "Doctor's report", provider = "Hospital", items = "Treatments", loss detail = "Diagnosis").
2. Add `<name>` to `case_types` in `config/settings.yaml`.
3. Put documents in `knowledge/<name>/policy/` and `knowledge/<name>/red_flags/`. Use Markdown with one heading per clause so each clause becomes its own passage, for example `### EXC-04 Commercial use`, or upload PDFs on the Knowledge page.
4. Put cases in `data/cases/<case_id>/`:
   - `statement.md`, `assessment_report.md`, `invoice.md`, and optionally `incident_report.md`
   - `case.json`, for example:
     ```json
     {"case_id": "H-0001", "case_type": "<name>", "policy_number": "P-1", "policy_start": "2024-01-15",
      "incident_date": "2024-06-01", "amount": 45000.0, "currency": "INR", "provider": "City Hospital"}
     ```
5. Optionally add past cases as lines in `data/past_cases.jsonl` with `past_case_id`, `case_type`, `incident_date`, `provider`, `amount`, `currency`, `outcome` (`paid`, `paid_reduced`, `rejected` or `sent_for_investigation`), `summary` and `note`.
6. Run `python run.py index`.

The `risk_model` section is optional. Leave it out if you have no labelled dataset, and the risk score shares that weight among its other parts. To add one, name a CSV, its label column and any columns to drop, then run `python run.py data`. Columns named `Sex` or `MaritalStatus` are never used by any model, whatever the file says.

## Adding real policy or warning-sign PDFs

On the **Knowledge** page choose a PDF, an insurance type and a kind (policy wording or warning signs), then press **Upload** and **Rebuild index**. You can also place files directly in `knowledge/<type>/policy/`, `knowledge/<type>/red_flags/` or `knowledge/general/red_flags/` and run `python run.py index`.

If a folder already holds your own documents, `python run.py data` will not copy the sample text into it. The answer key refers to clause IDs in the sample policy, so replacing that policy makes the coverage and warning-sign metrics meaningless for the sample cases.

## Sample data

**The case documents, past cases, policy wording and warning-sign guides are synthetic, written for this student project.** The sample knowledge texts each start with "SAMPLE TEXT FOR A STUDENT PROJECT – NOT A REAL POLICY." Their structure follows the usual definitions, cover, exclusions and conditions layout of a motor policy, but they are not any insurer's wording. The app shows a "Sample data" badge while sample cases are loaded.

Only the structured fields and fraud labels behind the risk model are real, from the dataset named below. Each sample case takes its date and risk-model fields from one row of that dataset; its story, amounts and documents are generated.

The 50 sample cases cover three insurance types, split by `case_type_shares` in `config/settings.yaml`: 15 vehicle, 18 health and 17 property (200 past cases split 60, 72 and 68). Only vehicle has a labelled dataset. **Health and property cases have generated dates and made-up fraud labels (`sample_fraud_rate`), and no risk model**, so their risk score is built from mismatches, warning signs and similar past cases only. Each type has its own labelled sample policy and warning-sign guide.

Generated documents have fixed sections (dates, items, amounts and totals) that come straight from the case's fact sheet, and narrative paragraphs written by the language model from that fact sheet. A narrative is checked for the required facts, a sensible length and invented numbers. If it fails twice, the scenario's template text is used instead. So planted mismatches and known answers are guaranteed to be in the documents.

## How each metric is produced

All metrics come from `tallyline/evaluate.py`, are saved as JSON in `results/`, and are copied into `results/RESULTS.md` without any other numbers. The answer key (`data/answer_key.jsonl`) is read only by the evaluator, never by the app.

| Metric | How it is produced | Data |
|---|---|---|
| ROC-AUC, PR-AUC, recall among the top tenth riskiest | 5-fold stratified cross-validation of logistic regression and random forest, on the dataset rows not used for sample cases | Real |
| Coverage clause found | Search the policy with the case's incident description; is each expected clause in the top passages? | Sample |
| Expected warning sign found | One search per mismatch and per statement sentence; is each expected sign in the results? | Sample |
| Look-alike past case found | Search past cases with the incident, place and provider; is the planted look-alike in the top 5? | Sample |
| Passage with known answer found | Search the case's documents with each known question; does a top passage contain the answer? | Sample |
| Answerable passing, and unanswerable stopped by `min_score` | Share of questions with at least one passage left, or none left, after the score threshold | Sample |
| Mismatch precision and recall, per type | Run the checks with the language model and compare with the planted mismatches | Sample |
| Coverage status accuracy | The model's status compared with the planted situation | Sample |
| Warning-sign precision and recall | Signs raised by the checks (see below) compared with the expected signs | Sample |
| Fact fields read correctly | Fields extracted by the model compared with the fact sheet | Sample |
| Known questions answered correctly | The answer must contain the known value | Sample |
| Unanswerable questions answered correctly | The answer must be exactly "Not found in the documents." | Sample |
| Note sentences kept and removed | Counted by the citation check on every note | Sample |

### How warning signs and the recommended action are decided

A document mismatch is already a checked fact, so it raises its closest warning sign without the language model. For the statement, the model picks the numbered sentences that look like a warning sign. It often names the wrong sign and sometimes picks an ordinary sentence, so the reranker chooses the sign and drops any sentence scoring below `red_flag_min_score`. That threshold was set on the sample wording, so real documents may need a lower one.

The recommended action follows fixed rules in `note.py`: excluded means reject; any mismatch or warning sign means investigate; unclear coverage or an unreadable document means ask for documents; otherwise pay. A wrong coverage status therefore gives a wrong recommendation.

To limit that, a coverage status must fit the clauses the model cited, using `clause_prefixes` in the case-type file. "Excluded" with no exclusion cited becomes "partly" if a limit was cited, "covered" if only a cover clause was cited, and otherwise "unclear". "Covered" with a limit cited becomes "partly". "Partly" with no limit cited becomes "covered" if a cover clause was cited, and otherwise "unclear". This cannot catch a status that fits the cited clause when the clause itself is the wrong one.

### Why questions about a case skip `min_score`

`min_score` still applies to policy and past-case searches. It does not apply to a case's own documents, because on a case added from PDFs fair questions scored 0.10 to 0.28 and unrelated questions 0.09 to 0.30, so no threshold separates them. A case has few passages, so the model reads them and says when the answer is not there, and the citation check still runs on what it says. The cost is one model call for every question, even an unanswerable one.

### How the citation check works

Every note sentence and answer must cite passages that were actually given to the model, and every number, date and reference ID in it must appear in the passages it cites. Anything else is removed and logged in the `citation_log` table. An answer that fails becomes "Not found in the documents."

The check confirms that the figures are supported by the cited text. It does not confirm that the sentence's meaning is right, so an investigator should still read the cited passages.

### What the risk score is

A weighted sum of the risk model's probability, the number of mismatches, the number of warning signs and the share of similar past cases that were rejected or investigated. The weights are in `config/settings.yaml` and shown in the UI. The score is a prompt for review, not a finding of fraud, and the logistic regression's balanced class weights mean its probability is a ranking signal rather than a calibrated chance.

## Licences

| Item | Source | Licence |
|---|---|---|
| Dataset titled "Vehicle Insurance Claim Fraud Detection" (`fraud_oracle.csv`) | [Kaggle, shivamb](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection) | CC0: Public Domain |
| Qwen2.5 3B Instruct | [Qwen on Hugging Face](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), packaged by Ollama | Qwen Research License: non-commercial use |
| Qwen2.5 7B Instruct (optional) | [Qwen on Hugging Face](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), packaged by Ollama | Apache 2.0 |
| bge-small-en-v1.5 embedding model | [BAAI on Hugging Face](https://huggingface.co/BAAI/bge-small-en-v1.5) | MIT |
| ms-marco-MiniLM-L-6-v2 reranker | [cross-encoder on Hugging Face](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) | Apache 2.0 |
| htmx | [htmx.org](https://htmx.org) | Zero-Clause BSD |

Check each licence at its source before any use beyond this student project. The 3B model in particular does not allow commercial use.
