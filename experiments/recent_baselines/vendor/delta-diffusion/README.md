# DELTA: Training-Free Character-Length Control in Summarization with Diffusion Language Models

Code for the paper **"Training-Free Character-Length Control in Summarization with Diffusion Language Models"**, accepted to the **EMNLP 2026 Main Conference**.

DELTA (**D**iffusion **E**xpected **L**ength **T**argeting **A**lgorithm) is a training-free, inference-time method that steers masked diffusion language models toward a target character length. At each denoising step, DELTA estimates the expected character length from both committed tokens and the probability distributions over remaining masks, then dynamically inserts or removes masked positions to steer the generation buffer toward the target.

DELTA requires no fine-tuning or architectural changes and can be applied to off-the-shelf masked diffusion language models.

## Results

DELTA reduces character-length errors by **an order of magnitude** compared to strong baselines while maintaining competitive summary quality.

### XSum

| Method          |     MAE |      <10 |       R-1 |       R-L |       BF1 |
| --------------- | ------: | -------: | --------: | --------: | --------: |
| **DELTA+LLaDA** | **2.3** |     99.6 |     0.286 |     0.217 |     0.882 |
| **DELTA+Dream** | **2.3** | **99.7** | **0.333** | **0.255** | **0.890** |
| LA-SF-AR        |     5.2 |     84.2 |     0.273 |     0.204 |     0.881 |
| GPT-4o-mini     |    13.1 |     48.4 |     0.264 |     0.194 |     0.880 |
| Lead-char-snap  |    15.7 |     60.5 |     0.166 |     0.122 |     0.850 |
| TextRank        |    14.5 |     50.1 |     0.178 |     0.130 |     0.845 |

### CNN/DailyMail

| Method          |     MAE |      <10 |       R-1 |       R-L |       BF1 |
| --------------- | ------: | -------: | --------: | --------: | --------: |
| **DELTA+LLaDA** | **2.4** | **99.6** |     0.363 |     0.259 |     0.876 |
| **DELTA+Dream** |     3.0 |     99.3 |     0.409 |     0.281 |     0.880 |
| LA-SF-AR        |    10.3 |     59.7 |     0.399 |     0.254 |     0.879 |
| GPT-4o-mini     |    96.6 |     11.1 |     0.393 |     0.245 |     0.877 |
| BART-CNN†       |   104.7 |      5.7 | **0.441** | **0.306** | **0.882** |
| Lead-char-snap  |    59.4 |     21.4 |     0.363 |     0.230 |     0.868 |
| TextRank        |    17.3 |     32.4 |     0.356 |     0.235 |     0.866 |

† BART-CNN is fine-tuned on CNN/DailyMail.

### arXiv

| Method          |     MAE |      <10 |       R-1 |       R-L |       BF1 |
| --------------- | ------: | -------: | --------: | --------: | --------: |
| **DELTA+LLaDA** | **2.8** | **98.4** |     0.443 |     0.234 |     0.837 |
| **DELTA+Dream** |     3.1 |     98.3 |     0.431 |     0.236 | **0.853** |
| LA-SF-AR        |    44.9 |     22.9 | **0.466** | **0.250** |     0.845 |
| GPT-4o-mini     |   112.2 |      3.7 |     0.432 |     0.217 |     0.839 |
| Lead-char-snap  |    96.5 |      7.0 |     0.351 |     0.183 |     0.837 |
| TextRank        |    46.9 |     12.0 |     0.406 |     0.212 |     0.837 |

**MAE** = mean absolute character-length error.
**<10** = percentage of outputs within 10 characters of the target.
**R-1/R-L** = ROUGE-1/ROUGE-L.
**BF1** = BERTScore F1.

## Additional Evaluation

The paper additionally reports:

* a fixed-buffer diffusion ablation;
* a comparison with DAEDAL;
* wall-clock runtime measurements;
* a standard Lead-3 diagnostic on CNN/DailyMail;
* G-Eval results using GPT-5-mini, with GPT-4o-mini results provided for comparison;
* a blind two-annotator human evaluation on 50 CNN/DailyMail examples.

In the human evaluation, DELTA+Dream, LA-SF-AR, and GPT-4o-mini receive average ratings of **4.27**, **4.32**, and **4.31**, respectively, on a 1--5 scale across relevance, consistency, coherence, and fluency.

## Installation

```bash
pip install transformers safetensors tokenizers huggingface-hub datasets tqdm accelerate

# Evaluation metrics
pip install rouge-score bert-score nltk
```

## Usage

### Run DELTA evaluation

Example for LLaDA on CNN/DailyMail:

```bash
python delta/eval_diffusion.py \
    --dataset cnndm \
    --model llada \
    --num-samples 500 \
    --confidence-threshold 0.65 \
    --low-confidence-threshold 0.90 \
    --confidence-metric probability \
    --decoder-mode delta \
    --estimate-avg-chars-per-token
```

Available datasets:

```text
cnndm
arxiv
xsum
```

Available models:

```text
llada
dream
```

### Key arguments

| Argument                         | Description                                                                        |
| -------------------------------- | ---------------------------------------------------------------------------------- |
| `--dataset`                      | Dataset: `cnndm`, `arxiv`, or `xsum`                                               |
| `--model`                        | Model: `llada` (LLaDA-8B-Instruct) or `dream` (Dream-7B-Instruct)                  |
| `--num-samples`                  | Number of examples to evaluate                                                     |
| `--confidence-threshold`         | $\tau$: threshold for committing/unmasking tokens                                  |
| `--low-confidence-threshold`     | $\tau_{\mathrm{low}}$: masks below this threshold are eligible for removal         |
| `--confidence-metric`            | Confidence metric, e.g. `probability`, `margin`, or `entropy`                      |
| `--decoder-mode`                 | `delta` for dynamic-buffer decoding or `fixed_buffer` for the ablation             |
| `--estimate-avg-chars-per-token` | Estimate characters per token from validation references for initial buffer sizing |

## Hyperparameters from the Paper

| Dataset      | Model | $\tau$ | $\tau_{\mathrm{low}}$ |
| ------------ | ----- | -----: | --------------------: |
| CNN/DM, XSum | Dream |   0.65 |                  0.95 |
| CNN/DM, XSum | LLaDA |   0.65 |                  0.90 |
| arXiv        | Dream |   0.90 |                  0.82 |
| arXiv        | LLaDA |   0.71 |                  0.71 |

## Run Baselines

### LA-SF-AR

The LA-SF-AR baseline uses Llama-3.1-8B-Instruct and requires access through Hugging Face.

```bash
python baselines/baseline_la_sf_ar.py \
    --dataset cnn_dailymail \
    --split test
```

### DAEDAL

DAEDAL (Chen et al., 2025) requires the DAEDAL repository to be available on `PYTHONPATH`.

```bash
python baselines/baseline_daedal.py \
    --dataset cnn_dailymail \
    --split test
```

### Fixed-buffer diffusion

The fixed-buffer ablation uses the same diffusion model and decoding setup as DELTA but disables dynamic mask insertion and removal:

```bash
python delta/eval_diffusion.py \
    --dataset cnndm \
    --model llada \
    --confidence-threshold 0.65 \
    --low-confidence-threshold 0.90 \
    --confidence-metric probability \
    --decoder-mode fixed_buffer \
    --estimate-avg-chars-per-token
```

## Compute Evaluation Metrics

```bash
python eval_metrics.py \
    --input results/cnndm/cnndm_results_llada.json \
    --method rouge bertscore length_metrics
```

## File Structure

```text
delta-diffusion/
├── delta/
│   ├── diffusion_lm.py           # DELTA algorithm and diffusion LM wrappers
│   ├── eval_diffusion.py         # Evaluation on CNN/DM, arXiv, and XSum
│   └── hparam_search.py          # Hyperparameter search
├── baselines/
│   ├── baselines.py              # BART-CNN, Lead-char-snap, TextRank, GPT-4o-mini
│   ├── baseline_la_sf_ar.py      # LA-SF-AR
│   ├── baseline_daedal.py        # DAEDAL
│   ├── eval_lead3_cnndm.py       # Standard Lead-3 diagnostic on CNN/DM
│   └── generate_lead_3.py        # Generate Lead-3 summaries
├── LlmAsJudge/
│   └── model.py                  # Local and OpenAI LLM-as-a-judge agents
├── results/                      # Evaluation outputs (not tracked in git)
├── eval_metrics.py               # ROUGE, BERTScore, and length metrics
├── eval_llm_as_judge.py          # LLM-as-a-judge evaluation
├── aggregate_results.py          # Aggregate timing and length-error statistics
├── aggregate_runs.py             # Aggregate multi-run metrics
└── compare_results.py            # Compare methods across datasets
```

## Models

DELTA is designed for masked diffusion language models. We evaluate it on:

* **LLaDA-8B-Instruct** — [GSAI-ML/LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) — Nie et al. (2025)
* **Dream-7B-Instruct** — [Dream-org/Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) — Ye et al. (2025)

## Repository

https://github.com/DFKI-NLP/delta-diffusion

## Citation

```bibtex
@inproceedings{castle2026delta,
  title     = {Training-Free Character-Length Control in Summarization with Diffusion Language Models},
  author    = {Castle, Steffen and Feldhus, Nils and Ebert, Christopher and Hennig, Leonhard and M{\"o}ller, Sebastian},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

The citation will be updated with the canonical ACL Anthology entry once available.


