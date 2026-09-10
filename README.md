# RaMamba: Rank-Guided Dual-Stream Mamba

Official implementation of **RaMamba: Rank-Guided Dual-Stream Mamba**.

RaMamba is a dual-stream model for long-term time series forecasting. It uses
the spectral-rank asymmetry between decomposed components to allocate model
capacity: a bidirectional variable-direction Mamba encoder processes the
higher-rank seasonal residual, while a channel-independent pooling MLP handles
the lower-rank trend.

## Key Features

- **Rank-guided routing**: Matches branch capacity to the measured spectral rank of each component.
- **Seasonal Mamba stream**: Models high-rank inter-variable dynamics with bidirectional variable-axis scanning.
- **Trend MLP stream**: Models smooth, low-rank trends with a lightweight pooling MLP.
- **Non-stationarity handling**: Combines RevIN with EMA/DEMA decomposition.
- **Controlled ablations**: Includes MLP-only, trend-Mamba, all-Mamba, and temporal-Mamba variants.

## Model Architecture

The core `RaMamba` model consists of:

1. **RevIN**: Reversible Instance Normalization to handle distribution shift.
2. **Decomposition**: Splitting input into Trend and Seasonality.
3. **Dual Streams**:
   - **Seasonal Stream**: Data Embedding (Inverted) + Mamba Encoder + Projector.
   - **Trend Stream**: Multi-layer MLP with Average Pooling and Layer Normalization.
4. **Fusion**: Linear concatenation of both streams to produce the final prediction.

## Getting Started

### 1. Environment Setup

It is recommended to use the `mamba` environment provided in this repository.

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Place your datasets (e.g., `ETTh2.csv`) in the `./dataset/` directory.

### 3. Training & Evaluation

You can reproduce the experiments using the provided shell scripts in the `scripts/` directory.

Run the main model with `--model RaMamba`. The original scripts continue to use
the legacy `DMamba` alias so previously reported commands remain reproducible.

#### Reproduce ETTh2 Seasonal Experiment

```bash
bash scripts/etth2/reproduce_etth2_DMamba_seasonal.sh
```

#### Other Ablation Experiments

```bash
# Example: Run AllMamba ablation
bash scripts/ablation/run_ablation_DMamba_AllMamba.sh
```

## Project Structure

- `models/`: Core RaMamba model and ablation variants. `DMamba` names are retained as compatibility aliases.
- `layers/`: Component layers (Mamba Encoder, Decomposition, Embedding, RevIN)
- `scripts/`: Shell scripts for reproducing experiments on various datasets (ETT, etc.)
- `exp/`: Experiment management logic (`exp_main.py`)
- `run.py`: Entry point for training and testing.
- `requirements.txt`: List of required Python packages.

## Results

The model outputs results (MSE/MAE) to the `results/` directory and logs to `logs/`.

## Citation

The camera-ready citation will be added after the ICONIP 2026 proceedings metadata is finalized.
