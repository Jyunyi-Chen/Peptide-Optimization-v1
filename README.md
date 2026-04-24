# Peptide Optimization v1

A reinforcement learning framework for multi-objective antimicrobial peptide (AMP) optimization using Proximal Policy Optimization (PPO).

## Overview

The agent iteratively mutates a seed peptide sequence to maximize therapeutic activity scores (ACP, AFP, AMP, AVP) while minimizing hemolytic toxicity (HEM). A reward engine combining physicochemical design rules and pretrained ML classifiers guides the search.

## Project Structure

```
peptide_optimization/   # PPO agent, environment, reward engine, encoding
amp_prediction/         # Antimicrobial peptide classifier (AI4AMP)
acp_prediction/         # Anticancer peptide classifier (AI4ACP)
afp_prediction/         # Antifungal peptide classifier (ensemble: Doc2Vec + PC6 + BERT)
avp_prediction/         # Antiviral peptide classifier (AI4AVP)
hem_prediction/         # Hemolysis classifier (LysisPeptica / PepBERT)
streamlit_app.py        # Interactive training dashboard
run_train.py            # CLI training entry point
config.py               # Hyperparameters and training settings
```

## Prediction Models

| Module | Method | Target |
|--------|--------|--------|
| AMP | AI4AMP (PC6 + CNN) | Antimicrobial activity |
| ACP | AI4ACP (PC6 + CNN) | Anticancer activity |
| AFP | Ensemble (Doc2Vec + PC6 + ProtBERT) | Antifungal activity |
| AVP | AI4AVP (PC6 + CNN) | Antiviral activity |
| HEM | LysisPeptica (PepBERT + CNN ensemble) | Hemolysis (minimize) |

> **Large model files are excluded from this repo.** Download them and place in the indicated paths before running AFP or HEM inference.
>
> | File | Size | Path | Download |
> |------|------|------|----------|
> | AFP ProtBERT checkpoint | ~1.6 GB | `afp_prediction/ensemble_model/bert/ensemble_prot_bert_bfd_epoch1_1e-06.pt` | [HuggingFace](https://huggingface.co/datasets/wccheng1210/AI4AFP_model/resolve/main/ensemble_model/bert/ensemble_prot_bert_bfd_epoch1_1e-06.pt?download=true) |
> | LysisPeptica HEM model | ~97 MB | `hem_prediction/lysispeptica_models_thr10/p843_750_5041chatt_ugml2std.keras` | [GitHub](https://github.com/lsbnbiis/LysisPeptica/blob/main/models/thr10/p843_750_5041chatt_ugml2std.keras) (open page → click **Raw** to download) |

## Requirements

- Python 3.12.13
- CUDA-capable GPU recommended

```bash
pip install -r requirements.txt
```

## Usage

### Train via CLI

```bash
python run_train.py
```

Configuration is in [config.py](config.py). Logs and checkpoints are saved to `peptide_optimization/logs/`.

### Train via Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard allows real-time monitoring of reward curves, peptide sequences, and prediction scores during training.

## Key Hyperparameters (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TARGET_PEPTIDE` | `RVKRVWPLVIRTVIAGYNLYRAIKKK` | Seed peptide for optimization |
| `N_EPISODES` | 100,000 | Training episodes |
| `TIME_HORIZON` | 5 | Steps per episode |
| `N_PARALLELS` | 200 | Parallel environments |
| `ENCODING_SCHEME` | `PepBERT-large` | Sequence embedding method |
