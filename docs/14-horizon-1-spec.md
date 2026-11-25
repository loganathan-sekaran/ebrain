# Horizon 1 Specification: Growth + Continual Learning

- **Status**: Draft for review (TRK-12, TRK-13). Revised November 25, 2026 after self-review and an independent review ([13 §11.1](13-design-review-2026-11.md#111-self-review-of-the-p0-changes-defects-found-and-fixed)), then extended with the second-pass recommendations R2, R3, R4 and R6 ([13 §11.2](13-design-review-2026-11.md#112-second-pass-design-review-recommendations-applied)).
- **Last updated**: November 25, 2026
- **Schedule, gate and budget**: [PLAN.md §3.1, §4](PLAN.md). This doc gives the *how*: protocol and implementation parameters, and the per-run budget breakdown. Gate thresholds and totals live only in PLAN.md, which wins on any conflict.

---

## 1. Hypothesis and Novelty

**Hypothesis.** On a class-incremental task stream, take a network that adds **function-preserving residual blocks** *when a growth controller detects capacity pressure*. Protect its existing parameters with online EWC and replay. Compared with the same final architecture present from the start (the *pre-allocated* control) and trained the same way, it will:

- reach higher average accuracy;
- retain old tasks no worse;
- and the *timing* of growth will matter. Growth at random feasible times, with the same number of events, will do worse. Growth on a fixed schedule at task boundaries will do no better.

**Relation to prior work.** Triggered growth is not new in itself:

- **Plateau-triggered growth** goes back to dynamic node creation (Ash 1989) and Cascade-Correlation (Fahlman & Lebiere 1990).
- **In continual learning**, CPG (Hung et al. 2019) grows when an accuracy goal is unmet. DEN (Yoon et al. 2018) grows by a sparsity rule. Learn-to-Grow (Li et al. 2019) and RCL (Xu & Zhu 2018) learn the growth decision. Progressive Nets and DER (Yan et al. 2021) grow per task.
- **Function-preserving growth** includes Net2Net, GradMax (Evci et al. 2022) and Firefly (Wu et al. 2020). Self-Expanding Neural Networks (Mitchell et al. 2023) decide *when* to add capacity from a natural-gradient criterion.

**Growth or plasticity?** Freshly added, unregularized parameters may help simply by restoring plasticity, the ability to keep learning, which deep networks lose over long task sequences (Dohare et al., *Nature* 2024) and which shows up as primacy bias in RL (Nikishin et al. 2022). H1 therefore also runs cheaper plasticity fixes, continual backprop and shrink-and-perturb, on the pre-allocated architecture (§3.3), and reports learning accuracy task by task.

**What H1 tests** is narrower: whether a *CL-aware* pressure signal (a plateau, plus underfit **or** replay interference) improves class-incremental accuracy when *timing is the only difference*. That is, against the same final architecture pre-allocated, against random feasible timing, and against boundary timing, under the same CL protection, and with memory reported in bytes (MEMO, Zhou et al. 2023).

Either answer is publishable as a careful empirical result. If Gate 1 condition 4 or 5 fails, the controller claim is dropped; see [PLAN.md §6.1](PLAN.md#61-adaptation-triggers).

---

## 2. Tracks

| Track | Backbone | Trainable | Growth unit | Purpose |
|---|---|---|---|---|
| **A: from scratch** (gated) | Reduced ResNet-18 (nf = 20, widths 20/40/80/160, ≈1.09M parameters), with a dropout layer (p = 0.1) before the head in **every** method | Everything | Residual BasicBlock (§4.1) | Tests the growth claim without help from pretraining |
| **B: frozen features** (reported, not gated) | CLIP ViT-B/32 image encoder, frozen (512-d features, precomputed) | MLP trunk 512→512→512 (≈525k) + head | Residual MLP block, hidden 128 (≈133k) | Tests whether growth helps on top of a frozen scaffold, the regime H2–H3 use. It is compared against **nearest-class-mean (NCM) on the same features**, a strong baseline that cannot forget, and against the pre-allocated trunk. The result decides whether H2–H3 grow on top of frozen encoders, by the rule in [PLAN.md §3.1](PLAN.md#31-gate-1-end-of-month-3). |

---

## 3. Protocol

### 3.1 Streams

| Stream | Tasks | Classes per task | Train images per task (before val split) | Epochs per task | Steps per task |
|---|---|---|---|---|---|
| Split-MNIST | 5 | 2 | ≈11.3k–13.0k (depends on the digit pair) | 1 | ≈315–365 |
| Split-CIFAR-100 | 10 | 10 | 5,000 | 5 | ≈705 (141 per epoch) |

- **Batches.** Each optimizer step uses **32 current-task samples**, plus **32 replay samples** from task 2 on, for methods that replay.
- **Splits.** Every method trains on the same 90% of each task's training data. The other 10% is that task's validation slice, used only by the controller and for tuning.
- **Augmentation.** CIFAR uses random crop (pad 4) and horizontal flip, applied to replayed samples too. MNIST has no augmentation. This is the same for every method.
- **Optimizer.** SGD, momentum 0.9, **constant learning rate within each task** (no scheduler). Learning rate and weight decay are tuned per method (§3.5).
- **Scenarios.** **Class-incremental** (one expanding head, no task ID at test time) is gated. **Task-incremental** (task ID given, multi-head) is reported alongside.
- **Seeds.** A seed fixes both the class-to-task assignment and the initialization, so every method sees the same streams and results are **paired by seed**.
  - Gate comparisons use seeds 0–9.
  - All other runs use seeds 0–4.
  - The gate seed count may be raised, never lowered, after the Month 2 pilot measures the variance of paired differences.
- **Replay budget.** 2,000 exemplars for CIFAR-100 and 500 for MNIST (`benchmark.replay_buffer_size`), identical for every replay method.
  - Reservoir sampling: each current-task sample is offered to the buffer once, on its first presentation.
- **Memory budget** ([PLAN.md §1](PLAN.md#operating-regime)). H1's training memory is the replay buffer plus the method's CL state. The CIFAR buffer is about 6.1 MB as uint8. All Gate 1 arms (full, pre-allocated, random-time, boundary) have identical buffers and the same final CL state by construction. DER runs at equal total memory. Every other method's memory is reported in bytes (§3.2), and a replay-budget sweep shows how the result depends on the budget (§3.4).
- **Compute budget.** Every method gets the same optimizer steps per task. FLOPs per sample are reported, since growth changes them.
- **Heads.** Every class-incremental method uses the same expanding linear head, with no bias correction except where a method defines its own (ER-ACE). Head parameters are excluded from the growth budget.

### 3.2 Metrics

R_{j,i} is test accuracy on task i after training through task j (class-incremental: full head; task-incremental: task i's head). T is the number of tasks. S_i is the test accuracy on task i of a network trained on task i alone, with the same budget.

| Metric | Definition | Source |
|---|---|---|
| ACC | (1/T) Σᵢ R_{T,i} | GEM (Lopez-Paz & Ranzato 2017) |
| BWT | (1/(T−1)) Σ_{i<T} (R_{T,i} − R_{i,i}) | GEM |
| Forgetting | (1/(T−1)) Σ_{i<T} max_{l<T} (R_{l,i} − R_{T,i}) | Chaudhry et al. 2018 |
| Learning accuracy (plasticity) | (1/T) Σᵢ R_{i,i}, also plotted per task i to show whether plasticity decays along the stream | Riemer et al. 2019 |
| FWT_ST | (1/(T−1)) Σ_{i>1} (R_{i,i} − S_i), task-incremental only. This deliberately uses a single-task reference. GEM's FWT compares against a *randomly initialized, untrained* model instead. | — |
| Params | Backbone + grown units at the end of the stream (head excluded) | — |
| Memory | Total bytes: parameters + CL state (Fisher, θ*, logits stored by DER++) + exemplars (uint8). Reported for every method, following MEMO's equal-memory argument. | MEMO (Zhou et al. 2023) |
| Compute | Training GPU-hours; inference FLOPs per sample | — |
| Calibration | From the uncertainty service (§4.5), on the class-incremental test set of all tasks seen so far, after every task: **ECE** (15 equal-width bins, after temperature scaling fit on the validation slices), **misclassification AUROC** (maximum softmax probability as the score), and **AURC** (area under the risk–coverage curve). Also on the validation slice just before and just after each growth event. | Guo et al. 2017; Hendrycks & Gimpel 2017; Geifman et al. 2019 (AURC) |
| Growth log | Events, steps and stages, and the trigger signal values at each check | — |

**Statistics.** Gate decisions use the paired lower-bound rule in [PLAN.md §3](PLAN.md#3-horizons-gates-and-benchmarks). Every table reports the mean and a 95% CI over seeds. Paired differences against the pre-allocated control are reported for all growth variants.

### 3.3 Baselines

All baselines run in the same harness, with the same backbone (including the head dropout), epochs, augmentation and replay budget.

| Baseline | Type | Scenario |
|---|---|---|
| Fine-tune | Lower bound | both |
| Joint (all tasks at once) | Upper bound | both |
| Online EWC, SI | Regularization | both |
| ER, A-GEM, DER++, ER-ACE | Replay (ER-ACE addresses class-incremental head bias) | both |
| DER (Yan et al. 2021), at equal total memory | Per-task expansion | class-incremental, reported |
| PackNet, Progressive Nets | Parameter isolation / growth | task-incremental only |
| NCM on frozen CLIP | Track B reference | both |
| **Pre-allocated static**: for each seed, the grown run's *final* architecture instantiated at task 1 (same zero-initialized last-BN γ), with the same CL method | **Primary control** (Gate 1 conditions 1–2) | both |
| **Width-matched static**: stage 4 widened so the parameter count matches the mean final size of the grown runs within ±1% (one stage-4 channel ≈ 9.5k parameters) | Shape control, reported | both |
| **Continual backprop** (Dohare et al. 2024) on the pre-allocated architecture, with the same CL method. Reinitialized units get their Fisher entries reset, like new parameters after growth (§4.4). | Plasticity control, reported ([PLAN.md §3.1](PLAN.md#31-gate-1-end-of-month-3)) | class-incremental |
| **Shrink-and-perturb** (Ash & Adams 2020) on the pre-allocated architecture, with the same CL method, applied at each task boundary after the EWC snapshot. Shrink factor and noise scale are tuned (§3.5). | Plasticity control, reported | class-incremental |

**Reproduction check.** Implementations may come from Avalanche or Mammoth, or be written by us. Before any implementation is trusted, it must reproduce a published reference number to within 3 points on that reference's own configuration. Record the reference and our number in the Gate 1 report.

### 3.4 Ablations (Track A, class-incremental Split-CIFAR-100)

The main results table covers both tracks and both scenarios. Ablations run only on this configuration.

| Variant | Tests |
|---|---|
| Growth + EWC + replay (**full**, primary trigger) | — |
| No growth (initial size) | Whether growth helps at all |
| Pre-allocated static | Growing vs having the same capacity |
| Width-matched static | Architecture shape |
| Growth, no EWC | EWC contribution |
| Growth, no replay | Replay contribution |
| **Random-time growth**: the same number of events per task as the full run with the same seed, at times drawn uniformly from that task's *feasible* decision checks (after warm-up, outside cooldown). Three draws per seed, averaged per seed. | Trigger timing (Gate 1 condition 4) |
| **Boundary growth**: the same total number of events as the full run with the same seed, one per task boundary, spread as evenly as possible over tasks 2–10 | Trigger vs fixed schedule (Gate 1 condition 5) |
| Underfit-only trigger (no interference term) | Value of the interference signal |
| Uncertainty trigger (§4.2) | Choice of trigger signal |
| **Replay-budget sweep**: full and pre-allocated at 500 and 5,000 exemplars (5 seeds each), next to the main 2,000 | Whether the growth effect depends on the memory budget ([PLAN.md §1](PLAN.md#operating-regime)) |

### 3.5 Tuning

- **Tuning stream.** Split-CIFAR-100 with class-order seed 1000. It uses the same images and classes as the evaluation stream, in a different task assignment. Decisions use only the validation slices.
- **Per-method grid.** Learning rate ∈ {0.01, 0.03, 0.1}, weight decay ∈ {0, 5e-4}, plus each method's own key hyperparameter: for example EWC λ ∈ {10, 100, 1000}, continual backprop's replacement rate, and shrink-and-perturb's shrink factor and noise scale. That is about 6–18 runs per method.
- **Controller thresholds** (δ, ρ, α, τ_H in §4.2) are tuned on the same stream, then frozen with the configs before the gate sweep.

---

## 4. Growth Mechanism

### 4.1 Growth unit: zero-initialized residual block

- **Track A.** Append a BasicBlock (two 3×3 convolutions with BatchNorm, identity shortcut, stride 1, d = the stage's width) to the end of stage k.
  - **The last BatchNorm's γ and β are initialized to zero**, so the residual branch outputs exactly 0. The block's input is already post-ReLU, so `relu(x + 0) = x`. Insertion is exactly function-preserving in both train and eval mode.
  - Block sizes are about 461k parameters at stage 4 (d = 160), 116k at stage 3, 29k at stage 2 and 7.3k at stage 1. Against an initial backbone of 1.09M, two stage-4 blocks take the network to ≈1.84×, so the 2× cap binds.
- **Track B.** Append a residual MLP block `y = x + W_up · GELU(W_down · LN(x))` to the trunk, with hidden size 128 and **`W_up` and its bias zero-initialized**. That is ≈133k parameters, so at most 3 blocks fit under the cap.
- **Stage choice.** Choose the deepest stage that has fewer than `max_blocks_per_stage` (2) added blocks **and** where `params + unit ≤ cap`. If no stage qualifies, the controller stops growing and logs a no-op. Choosing the stage by per-stage gradient norm is a stretch ablation.
- Width growth (adding neurons to existing layers) is **out of scope for H1**.

### 4.2 Trigger

All timing is in **fractions of the current task's optimizer steps**, so the same rules work on both streams.

| Parameter | Value |
|---|---|
| Check interval | every 5% of the task |
| Warm-up (checks run, no decisions) | first 20% of the task |
| Plateau window W | 3 checks |
| Cooldown after an event | 20% of the task |
| Max events per task | 2 |
| Growth in task 1 | none (no reference values exist yet) |
| EMA | over successive checks, smoothing 0.5, **reset at every task boundary** |

Signals, computed at every check:

```
val_ema       EMA of current-task validation loss (10% slice)
rel           (val_ema[t−W] − val_ema[t]) / val_ema[t−W]
plateau       0 ≤ rel < δ                                            δ = 0.01   (rising loss is NOT a plateau)
underfit      acc_ema < α · mean(final acc_ema of previous tasks)    α = 0.95   (acc on the current-task part of training batches, full head)
interference  replay_ema[t] > (1 + ρ) · replay_ema[first check of this task]    ρ = 0.10   (loss on replay batches already drawn; no extra data)
feasible      after warm-up ∧ outside cooldown ∧ events_this_task < 2 ∧ some stage qualifies (§4.1)

grow          feasible ∧ plateau ∧ (underfit ∨ interference)
```

- **Underfit-only variant**: `grow = feasible ∧ plateau ∧ underfit`.
- **Uncertainty variant**: replaces `(underfit ∨ interference)` with `H̄ > τ_H`. Here H̄ is the mean predictive entropy from the uncertainty service (§4.5) on the current-task validation slice. τ_H is set on the tuning stream as the 90th percentile of per-check H̄.
- **Boundary spikes.** A loss spike at a task boundary cannot trigger growth: the EMA resets, decisions wait for the warm-up, and rising loss fails `plateau`.

### 4.3 Optimizer handling

- On growth, call `optimizer.add_param_group({"params": new_params, "lr": lr, "weight_decay": wd})`. The new group's momentum buffer starts at zero. Its learning rate warms up linearly over 5% of the task.
- Existing parameter groups and their state are left unchanged. With a constant learning rate per task there is no scheduler to re-bind.

### 4.4 Online EWC with growth

- At the end of each task, compute the diagonal empirical Fisher on 1,000 samples drawn from the current task plus the replay buffer.
- Update `F ← γ·F + F_task`, with γ tuned within {0.9, 1.0}.
- Snapshot `θ* ← θ`.
- Penalty: `(λ/2) Σ_p F_p (θ_p − θ*_p)²`.
- **Parameters created after the last snapshot have no Fisher entry, so their penalty is zero** until the next boundary. The pre-allocated control isolates how much of the gain comes from this burst of unregularized plasticity.
- Keys are stable module paths. Blocks are appended to a `nn.ModuleList` per stage, which never renames existing entries.
- The Fisher and θ* cost memory: about 8.7 MB at 1.09M parameters, more than the 2,000-exemplar buffer (about 6.1 MB as uint8). This counts toward **Memory** (§3.2).

### 4.5 Uncertainty service

One component serves every consumer of uncertainty: the uncertainty trigger variant here, and later concept novelty, halting, escalation, curiosity and self-assessment ([02 C3](02-architecture.md#c3-uncertainty-service)).

- **H1 estimator**: MC dropout through the head dropout (10 passes), with temperature scaling. The temperature is refit on the validation slices of the tasks seen so far, at each task boundary and after each growth event.
- **Outputs**: calibrated class probabilities, predictive entropy, and a scalar confidence (the maximum calibrated probability).
- **Evaluation**: the calibration metrics in §3.2. A consumer may rely on the service only for quantities those metrics cover.

---

## 5. Implementation Constraints

- **API**:
  - `GrowableNetwork.grow(stage: int) -> GrowthEvent`.
  - **Never override `nn.Module.add_module`.** PyTorch uses it internally.
  - `GrowthController.check(step: int, signals: GrowthSignals) -> GrowthDecision` is a pure function of its inputs, so it can be unit-tested.
  - `UncertaintyService.predict(x) -> Prediction` (probabilities, entropy, confidence) and `UncertaintyService.calibrate(loader) -> float` (returns the fitted temperature).
- **Training loop**: plain PyTorch, single GPU, no DDP and no Lightning in H1. Both assume a fixed parameter set.
- **Checkpoints** store an *architecture manifest* (blocks per stage) next to the `state_dict`. To load, rebuild from the manifest first, then call `load_state_dict`. Load with `torch.load(..., weights_only=True)` or use safetensors.
- **Logging**: one JSONL event log per run (checks with all signal values, growth events, task boundaries, metrics), plus an optional W&B mirror.
- **Config**: a Hydra tree under `configs/`: `model/`, `growth/`, `cl_method/`, `optimizer/`, `benchmark/`, plus `seed` and `scenario`. Each value lives in exactly one place. For example, the replay size is `benchmark.replay_buffer_size`, and `cl_method` interpolates it.
- **Required unit tests** (CI). These are to be written with the code; only #7 exists today.
  1. Output is identical (to within `1e-6`) before and after `grow()`, in both train and eval mode.
  2. New parameters appear in the optimizer after growth.
  3. The EWC penalty is exactly 0 for parameters created after the last snapshot.
  4. The controller never grows during task 1 or the warm-up, inside the cooldown, past the cap, past the per-stage limit, or on a rising-loss or cross-boundary sequence.
  5. The same seed produces the same metrics on CPU (determinism).
  6. A checkpoint round-trip works with a grown architecture.
  7. The trigger schedule is feasible: every benchmark config gives at least `max_events_per_task` feasible decision checks per task (`tests/test_config.py`).
  8. ECE, misclassification AUROC and AURC match hand-computed values on a small example, and temperature scaling never changes the predicted class.

---

## 6. Compute Budget

| Item | Runs | GPU-h per run | GPU-h |
|---|---|---|---|
| Baseline reproduction checks | ~8 | ~1 | ~8 |
| Tuning (§3.5) | ~120 | ~0.5 | ~60 |
| Track A main table: ~14 methods × 5 seeds × 2 scenarios | ~140 | ~0.5 | ~70 |
| Track A gate set (class-incremental, seeds 0–9): full, pre-allocated, width-matched, random-time × 3 draws, boundary. Net of overlap with the main table. | ~60 | ~0.5 | ~30 |
| DER at equal memory (heavier) | ~10 | ~2 | ~20 |
| Plasticity baselines: continual backprop and shrink-and-perturb, seeds 0–9, plus their tuning | ~40 | ~0.5 | ~20 |
| Replay-budget sweep (§3.4) | ~20 | ~0.5 | ~10 |
| Track B sweep (features precomputed once) | ~75 | ~0.1 | ~8 |
| Split-MNIST, all methods | ~75 | ~0.05 | ~4 |
| Pilots, reruns, contingency | — | — | ~140 |
| **Total** | | | **≈ 370 (within the PLAN cap)** |

The per-run figures are estimates for a 24 GB consumer GPU. Replace them with measured values after the Month 2 pilot, and update the total.

---

## 7. Month-by-Month

| Month | Deliverables |
|---|---|
| 1 | CL harness (streams, metrics, JSONL logging, Hydra config); baselines in §3.3 reproduced, except the static controls and the plasticity baselines, which need the grown runs' final architectures; CI green |
| 2 | `GrowableNetwork`, `GrowthController`, online EWC + replay, uncertainty service; all §5 tests passing. Pilot on Split-MNIST and on 3 CIFAR tasks: measured per-run cost, and measured variance of paired differences, which sets the gate seed count. Controller thresholds tuned and frozen; `prereg/gate-1` tag ([PLAN.md §3](PLAN.md#decision-rule-all-gates)). |
| 3 | Grown runs → pre-allocated and width-matched controls → plasticity baselines, random-time and boundary controls → full sweep (main table on both tracks, ablations per §3.4) → **Gate 1 report** |

## 8. Gate 1 Report Template

1. Results table: ACC, BWT, forgetting, learning accuracy, params and memory, mean ± CI, every method and variant, both scenarios and both tracks.
   - Learning accuracy per task, for full, pre-allocated and the two plasticity baselines.
   - Calibration after every task (ECE, misclassification AUROC, AURC), and before vs after each growth event.
   - The replay-budget sweep.
2. Each Gate 1 condition marked pass or fail, with the paired difference, its 95% CI, and the margin.
3. Growth timeline plots: when each event fired, and the trigger signals at every check.
4. Reproduction checks: reference number vs ours, for each baseline.
5. Measured compute and memory vs budget.
6. Decision: **Go / No-Go**, and which adaptation trigger applies if No-Go.
7. Track B outcome and the scope decision it implies for H2–H3, by the rule in [PLAN.md §3.1](PLAN.md#31-gate-1-end-of-month-3).
8. Plasticity: whether continual backprop or shrink-and-perturb matches triggered growth, and what that means for the claim.
9. The `prereg/gate-1` tag, and any deviation from it.
10. The outcome against the priors in [PLAN.md §6.2](PLAN.md#62-risk-register).
