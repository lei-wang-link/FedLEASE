# FedLEASE: Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning

**Accepted by NeurIPS 2025**

**Authors:** Lei Wang*, Jieming Bian*, Letian Zhang, Jie Xu  
(*Equal contribution)

---

## Overview

FedLEASE is a federated learning framework that adaptively allocates and selects LoRA (Low-Rank Adaptation) experts for efficient fine-tuning of large language models across heterogeneous clients.

### Key Features

- **Multi-task Support**: Handle heterogeneous tasks (SST-2, QNLI, MRPC, QQP, etc.) across clients
- **Adaptive Expert Allocation**: Automatically determine optimal number of LoRA experts
- **Dynamic Routing**: Learnable routing mechanism for expert selection

### Three-Phase Approach

1. **Warmup Phase**: Clients train with single LoRA to learn task representations
2. **Clustering Phase**: Hierarchical clustering groups similar clients based on LoRA parameters
3. **Adaptive Training**: Clients use multiple LoRA experts with learnable routing

## Installation

```bash
# Clone repository
gh clone lei-wang-link/FedLEASE

# Install dependencies
pip install -r requirements.txt
```

**Important**: This project uses a **customized PEFT library** with multi-LoRA expert support. The modified `peft/` directory is included in the repository and will be used automatically.

---

## Quick Start

```bash
python main.py \
    --tasks sst2 qnli \
    --global_rounds 5 \
    --warmup_rounds 2 \
    --train_samples 100 \
    --test_samples 50
```

---


### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | roberta-large | Pre-trained model |
| `--tasks` | [sst2, sst2, ...] | Task list for each client |
| `--output_dir` | ./output | Output directory |
| `--global_rounds` | 25 | Number of federated rounds |
| `--local_epochs` | 2 | Local training epochs |
| `--warmup_rounds` | 5 | Warmup rounds before clustering |
| `--lr` | 3e-3 | Learning rate |
| `--rank` | 4 | LoRA rank |
| `--max_clusters` | 4 | Maximum LoRA expert clusters |
| `--train_samples` | 1000 | Training samples per client |
| `--test_samples` | 200 | Test samples per client |
| `--batch_size` | 128 | Training batch size |
| `--seed` | 42 | Random seed |

---

### Understanding Results

- **training_log.txt**: Round-by-round progress, clustering decisions
- **lora_client_map_round_X.json**: Client-to-expert group mapping
- **Visualizations**: Dendrograms, 2D embeddings, clustering metrics
- **training_history.json**: All evaluation metrics across rounds

---

## Project Structure

```
FedLEASE_code/
├── main.py              # Main entry point with CLI
├── client.py            # Client and WarmupClient classes
├── server.py            # Server aggregation logic
├── utils.py             # Dataset partitioning and clustering
├── peft/                # Customized PEFT library
│   └── tuners/lora.py  # Multi-LoRA expert implementation
├── requirements.txt     # Python dependencies
├── example_usage.sh     # Usage examples
├── LICENSE              # MIT License
└── README.md            # This file
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wang2025adaptive,
  title={Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning},
  author={Wang, Lei and Bian, Jieming and Zhang, Letian and Xu, Jie},
  journal={arXiv preprint arXiv:2509.15087},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License.

