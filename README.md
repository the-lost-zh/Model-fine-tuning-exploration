# Model Fine-tuning Exploration

Efficient fine-tuning methods for Vision Transformers: comparison and improvement.

Course project for 《机器学习与深度学习》.

## Methods

| Method | Type | Trainable Params (ViT-B) | Description |
|--------|------|--------------------------|-------------|
| Full FT | Baseline | 100% | All parameters trained |
| Linear Probe | Baseline | ~0.1% | Only classifier head |
| BitFit | PEFT | ~0.08% | Only bias terms |
| LoRA | PEFT | ~0.3-0.5% | Low-rank weight decomposition |
| SSF | PEFT | ~0.4% | Scale & shift feature modulation |
| AdaptFormer | PEFT | ~0.5-0.8% | Parallel bottleneck adapter |
| SSF-Sparse | Innovation | ~0.5% | Gated SSF with L1 sparsity |

## Setup

### Environment

```bash
conda create -n finetune python=3.11 -y
conda activate finetune
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install timm tqdm matplotlib seaborn pyyaml scikit-learn
```

### If HuggingFace is blocked (GFW)

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Or download weights manually and pass `--pretrained_path /path/to/weights.pth`.

### Datasets

Datasets are downloaded automatically via torchvision on first use:

```bash
python -c "
import torchvision.datasets as d
d.CUB200(root='./data', train=True, download=True)
d.Flowers102(root='./data', split='train', download=True)
d.StanfordCars(root='./data', split='train', download=True)
"
```

## Usage

### Single experiment

```bash
python main.py --method ssf --dataset cub200 --seed 42
python main.py --method lora --dataset flowers102 --lr 1e-3
python main.py --method ssf_sparse --dataset cub200 --sparsity_lambda 1e-5
```

### Sample efficiency experiment

```bash
python main.py --exp sample_efficiency --method ssf --dataset cub200
```

### Layer ablation experiment

```bash
python main.py --exp layer_ablation --method ssf --dataset cub200
```

### Run all experiments in parallel (3 GPUs)

```bash
bash scripts/run_parallel.sh
```

### Generate all figures

```bash
python scripts/generate_figures.py
```

## Project Structure

```
├── main.py                    # CLI entry point
├── configs/base.yaml          # Shared config
├── src/
│   ├── models/                # Method implementations
│   ├── data/                  # Dataset loading
│   ├── trainers/              # Training loop
│   ├── experiments/           # Experiment runners
│   └── visualization/         # Plotting
├── scripts/
│   ├── run_parallel.sh        # Multi-GPU launcher
│   └── generate_figures.py    # Figure generation
├── results/                   # Output directory
└── report/                    # Final report
```

## References

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
- Zaken et al., "BitFit: Simple Parameter-efficient Fine-tuning", ACL 2022
- Chen et al., "AdaptFormer: Adapting Vision Transformers", NeurIPS 2022
- Lian et al., "Scaling & Shifting Your Features", NeurIPS 2022
