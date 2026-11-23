# Clinical LLM Hallucination Detection

Group course project: detecting hallucinated statements in clinical text with a fine-tuned
clinical BERT model, using the [MedHAL](https://huggingface.co/datasets/GM07/medhal) dataset.

## Results

Bio_ClinicalBERT fine-tuned as a binary classifier (factual vs. hallucinated):

| Model | Data | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **Bio_ClinicalBERT, fine-tuned (3 epochs)** | 20,000-statement stratified sample, 80/20 split | **0.909** | **0.901** | **0.922** | **0.912** |

Scores are on the held-out 4,000-statement test split. `Model_training.ipynb` also contains
earlier runs kept for comparison: the model before fine-tuning, and a run on a 5,000-statement
sample.

## Data

MedHAL has 819,608 labelled medical statements (each with optional context, a label, an
explanation and a flag for synthetic vs. real). Label balance is close to even (417,000 true /
402,608 false); 327,568 statements are synthetic. 239,330 rows have no context.

## Notebooks

| Notebook | What it does |
|---|---|
| `Data_exploration.ipynb` | Loads MedHAL from Hugging Face; shape, missing values, label and synthetic/real distributions |
| `Preprocessing.ipynb` | Cleans each row and joins statement, context and explanation into one `full_text` input; writes the full dataset to CSV/JSONL in 100,000-row chunks |
| `Model_training.ipynb` | Fine-tunes `emilyalsentzer/Bio_ClinicalBERT` (PyTorch, Hugging Face Transformers, AdamW, lr 2e-5, max 512 tokens) and evaluates it |

The notebooks were run on Google Colab with a GPU and read/write files on Google Drive, so the
paths at the top of each notebook need changing to run elsewhere.

## Related work in this repo

`Reduction-of-Hallucinations-main/` is a separate part of the group project: a two-stage pipeline
that fine-tunes a causal language model for medical summarisation with supervised fine-tuning
(SFT) and then Direct Preference Optimization (DPO) using LoRA, plus preprocessing, PHI redaction
and adversarial data generation. Its own README explains how to run it.

## Limitations

- **The input includes MedHAL's `explanation` field, which can reveal the label.** Explanations
  for true statements are often "The statement is factual.", and explanations for false ones state
  the correct fact. The scores above therefore probably overstate how well the model detects
  hallucinations from the statement and context alone. The fix is to rebuild `full_text` from
  `statement` and `context` only and retrain.
- Training and testing used the first 20,000 rows of the preprocessed file, not a random sample
  of all 819,608, so results may not hold across every source in MedHAL.
- One random 80/20 split; no cross-validation or confidence intervals.
