# CNN vs Vision Transformer for Image Classification

Machine Learning and Applications, Topic 7. Compares the inductive biases of CNNs (ResNet,
ConvNeXt) and Vision Transformers with respect to dataset size, data augmentation and compute cost.

**Team:** [names, student IDs, team leader]

**This repository covers the CNN part.** ViT part: [teammate's name and repository link]

> Status: in progress. Sections marked TODO are completed as the project advances.

## Installation

Requires Conda and Python 3.11. The commands below install the CUDA 12.8 build of PyTorch.

```bash
conda create -n cnn-vit python=3.11 -y
conda activate cnn-vit
pip install -r requirements.txt
python scripts/check_env.py
```

## Project structure

```text
configs/   One YAML config per experiment, inheriting from base.yaml
src/       Data pipeline, models, training loop, interpretation
scripts/   Command-line entry points
tests/     Data pipeline tests
docs/      Project documents
results/   Final tables and figures (committed)
data/      Datasets (downloaded, not committed)
outputs/   Per-run logs and checkpoints (not committed)
```

## Reproducing the results

TODO: list the exact commands once the scripts are implemented.

## Results

TODO: main table (mean ± std over 3 seeds) and hardware used.

## Credits

| Source | Used for | Our changes |
|---|---|---|
| [timm](https://github.com/huggingface/pytorch-image-models) | Pretrained models | TODO |

TODO: list every other public source, and state clearly which parts the team wrote or modified.

## References

1. He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), 2015.
2. Dosovitskiy et al., [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929), 2020.
3. Liu et al., [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545), 2022.
