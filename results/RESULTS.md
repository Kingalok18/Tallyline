# Tallyline results

Case documents, past cases and knowledge texts are synthetic sample data written for this project. The structured fields and fraud labels of the risk model come from the real dataset.

These insurance types have no dataset, so their dates and fraud labels are generated and they have no risk model: health, property.

Free text in the case documents was written by: llm.

## Risk model

5-fold cross-validation on the rows set aside for the risk model. Sex and MaritalStatus are not used.

| Metric | Value | n | Data |
|---|---|---|---|
| vehicle, logistic regression: ROC-AUC | 0.8052 | 15345 | real |
| vehicle, logistic regression: PR-AUC | 0.1616 | 15345 | real |
| vehicle, logistic regression: recall among the top tenth riskiest | 0.2886 | 15345 | real |
| vehicle, random forest: ROC-AUC | 0.8332 | 15345 | real |
| vehicle, random forest: PR-AUC | 0.2292 | 15345 | real |
| vehicle, random forest: recall among the top tenth riskiest | 0.3823 | 15345 | real |

## Retrieval

Each search keeps the top passages after reranking, as set in config/settings.yaml (past cases keep a longer list). Warning signs are searched once per mismatch and per statement sentence.

| Metric | Value | n | Data |
|---|---|---|---|
| Coverage clause found, with reranker | 1.0 | 53 | sample |
| Expected warning sign found, with reranker | 1.0 | 32 | sample |
| Look-alike past case found, with reranker | 1.0 | 8 | sample |
| Passage with known answer found, with reranker | 1.0 | 200 | sample |
| Answerable questions passing min_score, with reranker | 0.955 | 200 | sample |
| Unanswerable questions stopped by min_score, with reranker | 0.58 | 50 | sample |
| Coverage clause found, without reranker | 0.9623 | 53 | sample |
| Expected warning sign found, without reranker | 1.0 | 32 | sample |
| Look-alike past case found, without reranker | 1.0 | 8 | sample |
| Passage with known answer found, without reranker | 1.0 | 200 | sample |
| Answerable questions passing min_score, without reranker | 1.0 | 200 | sample |
| Unanswerable questions stopped by min_score, without reranker | 0.0 | 50 | sample |

## Checks

Mismatches were planted in the sample cases at a chosen rate: 0.6 of cases with a fraud label and 0.3 of the others. These rates do not describe real cases.

For mismatches and warning signs, n is the number planted, so a small n means a rough estimate.

| Metric | Value | n | Data |
|---|---|---|---|
| Mismatch billed_items: precision | 1.0 | 2 | sample |
| Mismatch billed_items: recall | 1.0 | 2 | sample |
| Mismatch date_order: precision | 1.0 | 2 | sample |
| Mismatch date_order: recall | 1.0 | 2 | sample |
| Mismatch early_incident: precision | 1.0 | 2 | sample |
| Mismatch early_incident: recall | 1.0 | 2 | sample |
| Mismatch incident_report: precision | 1.0 | 3 | sample |
| Mismatch incident_report: recall | 1.0 | 3 | sample |
| Mismatch invoice_total: precision | 1.0 | 2 | sample |
| Mismatch invoice_total: recall | 1.0 | 2 | sample |
| Mismatch loss_detail: precision | 1.0 | 2 | sample |
| Mismatch loss_detail: recall | 1.0 | 2 | sample |
| Mismatch provider_repeat: precision | 1.0 | 3 | sample |
| Mismatch provider_repeat: recall | 1.0 | 3 | sample |
| All mismatches: precision | 1.0 | 16 | sample |
| All mismatches: recall | 1.0 | 16 | sample |
| Coverage status accuracy | 0.46 | 50 | sample |
| Warning signs: precision | 0.9697 | 32 | sample |
| Warning signs: recall | 1.0 | 32 | sample |
| Fact fields read correctly | 0.9698 | 696 | sample |

## Answers and notes

| Metric | Value | n | Data |
|---|---|---|---|
| Known questions answered correctly | 1.0 | 200 | sample |
| Unanswerable questions answered "Not found in the documents" | 1.0 | 50 | sample |
| Note sentences kept by the citation check | 0.9474 | 190 | sample |
| Note sentences removed | 10 | 190 | sample |
| Answers removed by the citation check | 0 | 200 | sample |

Language model: qwen2.5:3b-instruct. Real calls in this run: 24; reused from the cache: 549; model time in seconds: 570.7.
