# Topic 7 study and coding guide

**Goal:** compare the inductive biases of CNNs (ResNet, ConvNeXt) and ViTs with respect to
dataset size, data augmentation and compute cost.

**Your scope: the CNN part.** A teammate handles the ViT. This repo contains CNN code only, but
the assignment is a joint comparison, so tables below that mention ViTs are team-level. Agree the
following with your teammate early, or the two sets of results cannot be compared fairly: dataset,
train/val/test split, fractions, seeds, augmentation levels, tuning budget and the `metrics.json`
format. You still need to understand the ViT theory (3.3): every member can be asked about it.

## 1. What you must deliver

| Requirement | Source | Where it goes |
|---|---|---|
| Fine-tune pretrained CNN and ViT on 1 dataset | Topic 7, minimum | Milestone 3 |
| Train a small model from scratch for comparison | Topic 7, minimum | Milestone 4 |
| Learning curves against % of training data | Topic 7, minimum | Milestone 5 |
| Grad-CAM and attention maps | Topic 7, minimum | Milestone 6 |
| Simple baseline, progress report at end of week 3 | Rule 4.1, step 4 | Milestone 2 |
| Explain gaps with the papers; ablation on 1+ component | Rule 4.1, step 5 | Milestone 5 |
| 3+ seeds with mean ± std; comparable tuning budget; stated search space | Rule 4.2 | All experiments |
| Record hardware, training time, parameter count | Rule 4.2 | `metrics.json` |
| Regular Git commits, README, requirements, one config per experiment | Rule 4.3 | Whole repo |
| Credit public code; state what you wrote or modified | Rule 4.3 | README |
| Declare AI use; every member can explain the code and formulas | Rule 4.4 | `docs/AI_USAGE.md` |
| *Extension:* CIFAR-10-C robustness or shape/texture bias | Topic 7, extension | Milestone 7 |
| *Bonus (+1):* improvement or new application | Rule 4.1, step 6 | Milestone 7 |

**Administrative steps:** register 2 topic preferences early (each topic takes at most 2
teams, first come, first served), then submit the 1-2 page proposal by the end of week 1
(`docs/PROPOSAL_TEMPLATE.md`).

## 2. Recommended experimental design

**Models.** Both pretrained CNNs use ImageNet-1k weights and have about 26-29M parameters. For a
fair comparison, suggest your teammate uses a similar-sized ViT with ImageNet-1k weights (ViT-S/16,
`vit_small_patch16_224.augreg_in1k`, has about 22M).

| Role | Model | timm name |
|---|---|---|
| Classic CNN | ResNet-50 | `resnet50.a1_in1k` |
| Modern CNN | ConvNeXt-Tiny | `convnext_tiny.fb_in1k` |
| Scratch CNN | SmallCNN (yours) | `small_cnn` |

Pretraining hides much of the inductive-bias difference, so comparing your scratch CNN with a
scratch ViT from your teammate gives the clearest evidence. Propose this to them.

**Dataset: CIFAR-10** (10 classes; 50,000 training and 10,000 test images, 32×32 px). With
`val_fraction: 0.1`, 45,000 images remain for training.
- Pretrained models run at 224 px, so images are upscaled. `SmallCNN` runs at the native 32 px.
- Pretrained models may be close to their best by 10% of the data (450 images per class), so the
  fractions include 1% and 5%, where differences are more likely to show.
- **Limitation to state in the report:** upscaled 32 px images contain little fine detail, and
  the pretrained and scratch models see different resolutions.
- The CIFAR-10-C robustness extension uses the same classes, so it fits this choice directly.

**Experiment size.** Your learning-curve grid is 3 models × 6 fractions × 3 seeds = 54 runs.

Measured on this laptop's RTX 3050 Ti (random images at 224 px, mixed precision, short test, so
it excludes data loading and validation):

| Model | Batch size | Images/s | Peak GPU memory | One full epoch |
|---|---:|---:|---:|---:|
| ResNet-50 | 32 | 174 | 1.88 GB | 4.3 min |
| ConvNeXt-Tiny | 32 | 134 | 2.46 GB | 5.6 min |
| ViT-S/16 (teammate) | 32 | 247 | 1.47 GB | 3.0 min |
| ResNet-50 | 64 | 27 | 3.52 GB | 28.0 min |

Use batch size 32: at 64, speed collapses. Other programs (browser, VS Code, display) used about
1.9 GB of the 4 GB before training; when training needs more than what is left, it spills into
system memory and slows down several times. An earlier ConvNeXt measurement gave only 44 images/s,
probably for this reason. Close other GPU-heavy programs and check `nvidia-smi` before long runs.
At 8 epochs, the learning-curve grid needs roughly 3-4 hours each for ResNet-50 and ConvNeXt-Tiny,
before data loading, validation and tuning. Kaggle or Colab are a fallback if needed. If you move any run, record the
hardware for it, and compare training times only between runs on the same GPU. Re-measure with real
data once your pipeline works.

## 3. Theory to study

Study in this order. Be able to write each formula from memory: every member can be asked
about them in Q&A, and method and mathematics are 20% of the mark.

### 3.1 Inductive bias: the central idea
An inductive bias is an assumption built into a model before it sees data.
- **CNN, locality:** each neuron sees a small neighbourhood.
- **CNN, weight sharing:** the same filter is applied everywhere, giving translation
  equivariance, f(shift(x)) = shift(f(x)), with pooling adding some invariance.
- **ViT:** much weaker bias. Only patch splitting and the MLP are local; spatial relations must
  be learned, and attention is global from the first layer.

**Your hypothesis:** strong bias means less data is needed but capacity to learn is more
constrained; weak bias means more data or stronger augmentation is needed. You test this.

### 3.2 ResNet (He et al., 2015)
Read sections 1-3. Very deep plain networks showed higher *training* error (degradation).
Residual blocks learn the change instead of the full mapping:

$$y = F(x, \{W_i\}) + x$$

If identity is optimal, the block can push F toward zero. **Know:** the bottleneck block,
why this eases optimisation, and how the receptive field grows with depth.

### 3.3 ViT (Dosovitskiy et al., 2020)
Read sections 1-3 and the data-size results in section 4. An image is split into N = HW/P²
patches, and each is linearly projected to dimension D:

$$z_0 = [x_{class};\, x_p^1E;\, \dots;\, x_p^NE] + E_{pos}$$

$$z'_\ell = \text{MSA}(\text{LN}(z_{\ell-1})) + z_{\ell-1}, \qquad
z_\ell = \text{MLP}(\text{LN}(z'_\ell)) + z'_\ell, \qquad y = \text{LN}(z_L^0)$$

Scaled dot-product attention is:

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Know:** why divide by √d_k, what the class token does, why position embeddings are needed,
the O(N²) attention cost, and the paper's claim about ImageNet vs ImageNet-21k. Your experiment
tests this claim at a much smaller scale, which you must acknowledge when comparing.

### 3.4 ConvNeXt (Liu et al., 2022)
Read the design roadmap in section 2. It modernises ResNet step by step using ideas from
Transformers: staged compute ratio, patchify stem, depthwise convolution, inverted bottleneck,
7×7 kernels, GELU, fewer activations, LayerNorm. **Key lesson for your discussion:** part of
"ViT beats CNN" came from training recipe and design choices, not attention itself. So do not
attribute every difference you find to inductive bias.

### 3.5 Training
- **Cross-entropy:** $L = -\sum_c y_c \log p_c$
- **AdamW:** weight decay is decoupled from the gradient update. Know how this differs from L2
  regularisation in Adam.
- **Warmup and cosine decay:** $\eta_t = \eta_{min} + \tfrac12(\eta_{max}-\eta_{min})(1+\cos(\pi t/T))$
- **Fine-tuning:** a new head with a higher learning rate and a backbone with a lower one. Linear
  probe (frozen backbone) vs full fine-tuning. Why BatchNorm behaves differently from LayerNorm
  with small batches and frozen layers.
- **Augmentation:** random crop and flip (basic); RandAugment, Mixup and CutMix (strong). They add
  invariances the model does not have built in, so expect ViTs to benefit more. Verify it.

### 3.6 Interpretation

**Grad-CAM** (Selvaraju et al., 2017; not in the topic's list, so cite it). For class c and
feature maps A^k of the last convolutional layer:

$$\alpha_k^c = \frac{1}{Z}\sum_i\sum_j \frac{\partial y^c}{\partial A^k_{ij}}, \qquad
L^c_{\text{Grad-CAM}} = \text{ReLU}\!\left(\sum_k \alpha_k^c A^k\right)$$

**Attention rollout** (Abnar & Zuidema, 2020; also cite it): with head-averaged attention A_ℓ,

$$\tilde{A}_\ell = 0.5\,A_\ell + 0.5\,I, \qquad R = \tilde{A}_L \tilde{A}_{L-1} \cdots \tilde{A}_1$$

Take the class token's row of R and reshape it to the patch grid. **Caveats for the report:**
the maps are not directly comparable. Grad-CAM is class-specific and gradient-based, while
rollout is class-agnostic. Neither proves causation.

## 4. Coding milestones

Each milestone gives you natural commits. After each, you should be able to explain every line.

### Milestone 0: Git and proposal (week 1)
Finish `README.md` setup, push the first commit, write and submit the proposal.

### Milestone 1: `src/utils.py`
Implement in order: `load_config`, `set_seed`, `count_parameters`, `get_hardware_info`,
`make_run_dir`. **Check:** loading `configs/resnet50_finetune.yaml` shows merged values from
`base.yaml`.

### Milestone 2: `src/data.py` and baseline (weeks 2-3)
1. Load the dataset and view a batch of images with labels.
2. Implement the stratified split and nested subsets. Complete `tests/test_data.py`.
3. Implement `build_transforms`, then `build_dataloaders`.
4. Run `scripts/baseline_logreg.py`. **Results go in your end-of-week-3 progress report.**

**Check:** run `pytest`, and confirm the baseline clearly beats random guessing (1/10 = 10%).

### Milestone 3: Fine-tuning loop (`src/engine.py`, `scripts/train.py`)
1. `build_model` (timm branch only) and `get_param_groups`.
2. `train_one_epoch`, `evaluate`, `build_scheduler`, `fit`.
3. **Sanity check:** overfit 1 small batch. If training loss does not approach 0, there is a bug.
4. Run one short ResNet-50 job, then ConvNeXt-Tiny.

**Check:** ~95%+ test accuracy on CIFAR-10 with 100% of the data is a reasonable expectation for
pretrained models. Far below that usually indicates wrong normalisation, resizing or learning rate.

### Milestone 4: Scratch model (`src/models.py`)
Implement `SmallCNN`, with a parameter count agreed with your teammate's scratch ViT. **Check:**
verify tensor shapes for each stage, and overfit one batch again.

### Milestone 5: Experiments (weeks 4-6)
1. `scripts/run_experiments.py` with resumable runs.
2. Learning-curve grid. Then augmentation study. Then ablation: frozen vs full fine-tuning, and
   pretrained vs random initialisation.
3. `scripts/summarize_results.py`: tables and figures.
4. Give every model the same tuning budget, e.g. the same 4-6 learning-rate/weight-decay trials
   chosen on the validation set. Record the search space.

**Questions the report must answer (combine your results with your teammate's):**
- Where do the curves cross, if at all? Does ViT fall faster than CNNs at low data?
- How does the gap change between scratch and pretrained models?
- Which model gains most from strong augmentation?
- Accuracy per training second, and per parameter?
- Where do results differ from the papers, and why (dataset, scale, resolution, epochs)?

### Milestone 6: Interpretation (`src/interpret.py`, `scripts/visualize.py`)
Grad-CAM for each CNN, on one fixed list of correct and misclassified test images. Share that list
with your teammate so their attention maps show the same images.

### Milestone 7: Extensions and bonus (optional)
- Robustness on CIFAR-10-C, or shape/texture bias.
- Bonus ideas: apply to a Vietnamese agricultural or medical dataset; or test whether a small
  convolutional stem helps the scratch ViT at low data, which directly probes inductive bias.

### Milestone 8: Finalise (weeks 7-8)
Report (8-12 pages), slides or 10-minute video, final README check on a clean environment.
Submit at least 2 days before the final exam; lateness costs 10% of the report mark per day.

## 4b. Implementation checklist for each stub

**Configs:** `base.yaml` holds shared starting values, not final ones. Experiment configs set
`inherit: base.yaml` and override only what differs. Options: `data.augmentation` = `none`, `basic` or `strong`; `model.freeze_backbone:
true` trains the head only. Scratch models use more epochs (start at 60) than fine-tuning.

**`src/utils.py`**
- `load_config`: read the YAML; merge recursively over its `inherit` file; apply overrides such as
  `{"seed": 1, "data.train_fraction": 0.1}`; return a dict.
- `set_seed`: seed Python, NumPy and PyTorch (CPU and CUDA). Decide on cudnn deterministic/benchmark.
- `count_parameters`: return (total, trainable).
- `get_hardware_info`: return GPU name, VRAM, CPU, and torch/timm versions.
- `make_run_dir`: create `outputs/<experiment>/frac<fraction>_aug<aug>_seed<seed>/` and save the
  resolved config there.

**`src/data.py`**
- `load_raw_datasets`: return (train, test) from `torchvision.datasets.CIFAR10`, using
  `train=True` and `train=False`, with `root` from the config and `download=True`.
- `make_train_val_indices`: stratified, and fixed for a given seed whatever the train fraction.
- `subsample_train_indices`: stratified and nested, so the 10% subset lies inside the 25% subset.
- `build_transforms`: read mean, std and input size from
  `timm.data.resolve_model_data_config(model)`. Never apply random augmentation when
  `is_train=False`. List each augmentation level's exact operations in the report.
- `build_dataloaders`: return (train, val, test, num_classes). On Windows, scripts need the
  `if __name__ == "__main__"` guard for workers.

**`src/models.py`**
- `SmallCNN`: conv blocks (conv, batch norm, ReLU, downsampling), global average pooling, linear
  classifier. Keep a reference to the last conv layer for Grad-CAM.
- `build_model`: `small_cnn` returns your class; otherwise use
  `timm.create_model(name, pretrained=..., num_classes=...)`. If frozen, train only
  `model.get_classifier()`.
- `get_param_groups`: separate head and backbone learning rates; no weight decay on biases or norm
  layers. Be ready to explain why.

**`src/engine.py`**
- `build_scheduler`: linear warmup, then cosine decay.
- `train_one_epoch`: return loss and accuracy; use `torch.autocast` and `GradScaler` when `amp` is
  on; support `grad_accum_steps`.
- `evaluate`: use `@torch.no_grad()` and `model.eval()`; return loss, accuracy and macro-F1.
- `fit`: each epoch, train, validate and append to `history.csv`. Save the best checkpoint by
  validation accuracy. Measure epoch time, total time and `torch.cuda.max_memory_allocated`. At the
  end, load the best checkpoint, test once, and save `metrics.json` with test metrics, parameter
  count, times and hardware.

**`src/interpret.py`**
- `grad_cam_for_cnn`: return (batch, H, W) heatmaps scaled to [0, 1]. Write it with hooks, or use the
  `grad-cam` package and credit it.
- `overlay_heatmap`: draw the heatmap over the un-normalised image.

**`scripts/`**
- `baseline_logreg.py`: flattened 32×32 pixels (no augmentation); fit sklearn `LogisticRegression`;
  choose C on validation; report test accuracy over 3 seeds to `outputs/baseline/metrics.json`.
- `train.py`: parse `--config` and `--set`; seed; build the model, then its dataloaders; call `fit`.
- `run_experiments.py`: `--study learning_curve` (configs × fractions × seeds), `augmentation`, and
  `ablation` (frozen vs full, pretrained vs random). Skip runs with an existing `metrics.json`.
- `summarize_results.py`: collect every `metrics.json`; compute mean ± std; save
  `results/summary.csv`. Plot the learning curve (log-scale x, std band), plus accuracy vs time and
  accuracy vs parameters.
- `visualize.py`: load each model's best checkpoint (100% data, same seed); save a grid of images
  and heatmaps to `results/`.

**`tests/test_data.py`**: train and validation do not overlap; subsets are nested; classes stay in
about the same proportion at each fraction.

## 5. Common mistakes
- **Using the test set to choose hyperparameters or epochs.** Use validation; test once at the end.
- **Wrong normalisation.** Pretrained models expect their own mean/std; read them from timm.
- **Non-nested subsets,** which add noise to the learning curve.
- **Unequal tuning effort,** which makes your favourite model win unfairly.
- **Comparing epochs instead of steps** at different fractions: 10% data with equal epochs means
  10× fewer updates. Decide how to handle this, and state your decision.
- **CUDA out-of-memory:** lower `batch_size`, raise `grad_accum_steps`, keep `amp: true`.
- **Windows dataloader errors:** keep the `__main__` guard, or set `num_workers: 0`.

## 6. Git routine
Commit small working steps, with messages that say what changed:

```text
feat(data): add stratified nested subsets
test(data): verify subsets are nested
feat(engine): add AMP training loop
exp: run learning curve for ResNet-50
docs: add week 3 baseline results
```

```powershell
git status
git add src/data.py tests/test_data.py
git commit -m "feat(data): add stratified nested subsets"
git push
```

- The instructor may review contribution history, so commit steadily over the weeks rather than
  all at once, and have each teammate commit with their own account.
- Never commit datasets or checkpoints; `.gitignore` already excludes them.
- Do commit final tables and figures in `results/`.

## 7. Report outline and marks

| Rubric item | Weight | Report section |
|---|---|---|
| Background and problem understanding | 15% | Introduction: approaches, pros/cons, your choice |
| Method and mathematics | 20% | Section 3 of this guide, in your own words |
| Experiments and analysis | 30% | Baselines, seeds, ablations, gaps with the papers |
| Code and reproducibility | 10% | Runs from the README, sources credited |
| Presentation | 25% | Option A (20 + 10 min) or B (10-min video) |
| Improvement / application | +1 | Milestone 7 |

Appendix: task allocation, each member's contribution (%), and the AI declaration.
