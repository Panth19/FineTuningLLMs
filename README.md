# FineTuningLLMs

FineTuningLLMs contains example code and notebooks to fine-tune sequence-to-sequence and text-generation models (Hugging Face Transformers) on a simple prompt→response dataset. It includes a script-style example and two Jupyter notebooks showing dataset preparation, training, and evaluation (ROUGE).

## Quick summary
- Purpose: show an end-to-end workflow to fine-tune transformer models (e.g., T5 / CodeT5 / other text2text models) using a minimal dataset format and evaluate outputs with ROUGE.
- Intended users: ML engineers and researchers who want a compact starting point to fine-tune a Transformer model on custom prompt→response pairs.

## Contents
- Fine-tuning LLM code v1.0.py — scripted example: dataset loading, tokenization, Trainer setup, training loop, evaluation with ROUGE, and sample generation.
- model-fine-tuning.ipynb — notebook with the same end-to-end flow (includes example outputs and training logs).
- fine-tuning-gemma-2-with-lora.ipynb — notebook (LoRA tuning example).
- README.md — this file.

## Stack / dependencies
- Language: Python 3.8+
- Libraries used: transformers, datasets, torch, nltk, matplotlib, pandas, tqdm, sacrebleu, rouge_score, wandb (optional)
- Install (as used in the notebooks):
  pip install transformers[sentencepiece] datasets sacrebleu rouge_score py7zr torch matplotlib nltk

## Dataset format
The code expects a plain text dataset where lines alternate between prompt and corresponding target output. Example (in a single .txt):
- line 1: prompt 1
- line 2: output 1
- line 3: prompt 2
- line 4: output 2
Notes:
- The script cleans excessive spaces and strips quotes/newlines.
- You can change the loader to use JSON/CSV or Hugging Face datasets if preferred.

## How to run (shortest path)
1. Clone the repo:
   git clone https://github.com/Panth19/FineTuningLLMs.git
   cd FineTuningLLMs
2. Install dependencies (virtualenv highly recommended):
   pip install transformers[sentencepiece] datasets sacrebleu rouge_score py7zr torch matplotlib nltk
3. Edit the script before running:
   - Open `Fine-tuning LLM code v1.0.py`.
   - Set model checkpoint: change model_ckpt = "any/text2text-generation/model/from/huggingface" to a real HF model (example: "t5-small" or "Salesforce/codet5-base").
   - Set dataset path: replace 'path/to/dataset.txt' with the path to your dataset file.
   - Optionally tune hyperparameters (max lengths, batch size, epochs) in the TrainingArguments block.
4. Run:
   python "Fine-tuning LLM code v1.0.py"
   (Or open and run the notebooks in Jupyter / Colab / Kaggle.)

## Notebooks
- model-fine-tuning.ipynb contains the same workflow with interactive cells and plotted histograms of token lengths and training logs. It also shows usage of wandb (if enabled) and example ROUGE evaluation outputs.
- fine-tuning-gemma-2-with-lora.ipynb demonstrates LoRA-based fine-tuning (if you want parameter-efficient tuning).

## Key implementation notes (from code/notebooks)
- Model loader:
  - For seq2seq: use AutoModelForSeq2SeqLM + AutoTokenizer.
  - For decoder-only models you would use AutoModelForCausalLM (the script includes a comment about this).
- Dataset splitting: the script splits data into train/validation by a percentage (test_percentage = 0.9).
- Token lengths: the notebooks compute token-length histograms to choose max_input_length and max_target_length.
- Evaluation: uses ROUGE via datasets.load_metric('rouge').
- Generation: sample generation uses a Hugging Face pipeline('text2text-generation') and model.generate with beam search; generation kwargs can be tuned.
- Training: Trainer (transformers) with DataCollatorForSeq2Seq; example hyperparameters are present (many are placeholders and should be adjusted for real runs).

## Suggested training settings / gotchas
- GPU strongly recommended for non-trivial models.
- For large models, reduce per_device_train_batch_size and increase gradient_accumulation_steps.
- The script sets long defaults (e.g., num_train_epochs = 100 / 500) — set these to realistic values for your dataset.
- If using a decoder-only model, switch to AutoModelForCausalLM and adapt tokenization/targets accordingly.
- Watch out for deprecated APIs in newer transformers (the notebooks show `as_target_tokenizer` usage and warnings).

## Examples / snippets
- Minimal modification to run:
  - Change model_ckpt:
    model_ckpt = "t5-small"
  - Provide dataset:
    with open('my_dataset.txt','r') as file:
        lines = file.readlines()
  - Run script:
    python "Fine-tuning LLM code v1.0.py"





---

If you want me to commit this README.md and/or add the images extracted from the notebooks, tell me which of the two to do and whether you prefer files added at docs/images/ or assets/images/. I can proceed and push a commit.
