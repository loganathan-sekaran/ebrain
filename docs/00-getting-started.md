# Getting Started: Horizon 1 Onboarding

- **Last updated**: November 25, 2026
- **Audience**: an engineer joining the project during Horizon 1 (H1)

## What E-Brain Is Testing Right Now

E-Brain is currently testing one claim on standard continual-learning (CL) benchmarks. Take a network that adds function-preserving residual blocks *when a growth controller detects capacity pressure*: a loss plateau, plus underfit or replay interference. Protect its existing parameters with online EWC and replay. The claim is that it beats the same final architecture present from the start (the *pre-allocated* control) on accuracy, without worse retention. The *timing* of growth should also matter: random feasible timing should do worse, and boundary timing no better. The full hypothesis and novelty statement are in [14-horizon-1-spec.md §1](14-horizon-1-spec.md#1-hypothesis-and-novelty).

> **Numbers live in one place.** Every schedule, gate threshold, benchmark, budget and team number is in [PLAN.md](PLAN.md). This document does not restate them. If anything here seems to disagree with PLAN.md, PLAN.md wins; please fix this file.

What is *not* on the H1 critical path: per-neuron models, spiking/STDP, neurogenesis, reward systems, language, vision encoders beyond the H1 backbones, and tools. Spiking/STDP work is a research branch, described in [Appendix A](appendix-a-bio-inspired-neuron-research-branch.md).

---

## 1. Reading Order

1. [PLAN.md](PLAN.md): horizons, gates, budget, terminology. Note that "Phase" means a *developmental* phase of the model, never a project step.
2. [14-horizon-1-spec.md](14-horizon-1-spec.md): the H1 protocol, metrics, baselines, ablations, growth mechanism and required tests. This is your working spec.
3. [02-architecture.md](02-architecture.md) §0: why the substrate is vectorized PyTorch, and the scaffolded hybrid architecture.
4. [13-design-review-2026-11.md](13-design-review-2026-11.md): what changed and why, especially §4.2 (growth mechanics that were wrong before) and §11.
5. [DECISIONS.md](DECISIONS.md): the context behind each decision on record, when you want to know why something is the way it is.
6. Others as needed: [02 §1](02-architecture.md#1-component-cards) for the component cards, [15](15-horizon-2-spec.md) and [16](16-horizon-3-spec.md) for the H2 and H3 spec drafts, [03](03-developmental-phases.md) for developmental phases, [Appendix A](appendix-a-bio-inspired-neuron-research-branch.md) for the spiking research branch. [Appendix B](appendix-b-architecture-v1-unscheduled.md) holds the old v1 component designs; do not build from it.

---

## 2. Environment Setup

Requirements: Python 3.12 or newer, Git, and (for real runs) one CUDA GPU. CPU is enough for unit tests and Split-MNIST smoke runs.

Repository layout:

| Path | Contents |
|---|---|
| `src/ebrain/` | The package. All library code goes here. |
| `configs/` | Hydra config tree: `model/`, `growth/`, `cl_method/`, `optimizer/`, `benchmark/`, plus `seed` and `scenario` |
| `tests/` | pytest suite. Today it holds the config, schedule-feasibility, docs-link and single-source tests; the other required H1 tests arrive with the code ([14 §5](14-horizon-1-spec.md#5-implementation-constraints)). |
| `docs/` | Design docs. PLAN.md is canonical. |

Clone the repository:

```bash
git clone https://github.com/loganathan-sekaran/ebrain.git && cd ebrain
```

Create a virtual environment:

```bash
python3.12 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install the package in editable mode with development extras:

```bash
pip install -e ".[dev]"
```

Install the git hooks:

```bash
pre-commit install
```

Lint and check formatting:

```bash
ruff check . && ruff format --check .
```

Type-check:

```bash
mypy src
```

Run the tests:

```bash
pytest
```

CI (`.github/workflows/ci.yml`) runs the same checks on Python 3.12 and 3.13, with a CPU-only PyTorch.

---

## 3. First Week

Aligned with [14 §7](14-horizon-1-spec.md#7-month-by-month), Month 1 (TRK-16).

- [ ] Environment set up; `ruff`, `mypy` and `pytest` pass locally.
- [ ] Read PLAN.md and doc 14 in full. Write down any question the spec does not answer and raise it before coding.
- [ ] CL harness skeleton in `src/ebrain/`:
  - [ ] Task streams: Split-MNIST and Split-CIFAR-100, with the class-to-task assignment fixed by seed ([14 §3.1](14-horizon-1-spec.md#31-streams)).
  - [ ] Class-incremental (single expanding head) and task-incremental (multi-head) scenarios.
  - [ ] Accuracy matrix R_{j,i} and the metrics in [14 §3.2](14-horizon-1-spec.md#32-metrics): ACC, BWT, forgetting, learning accuracy, FWT_ST, params, memory in bytes, compute. Calibration metrics come with the uncertainty service in Month 2.
  - [ ] Reservoir replay buffer with a configurable, method-independent budget.
  - [ ] One JSONL event log per run: task boundaries, metrics, and (later) growth events.
  - [ ] Hydra config tree under `configs/`.
- [ ] Reproduce one baseline end to end: **fine-tune** and **ER** on Split-MNIST, several seeds, ACC and BWT reported with 95% CIs.
- [ ] Unit tests for the metric functions on a hand-computed R matrix.
- [ ] CI green on your PR.

---

## 4. Horizon 1 Checklist

This mirrors [14 §5](14-horizon-1-spec.md#5-implementation-constraints) and [14 §7](14-horizon-1-spec.md#7-month-by-month). Follow the spec for details; do not duplicate it here.

### Month 1 of Horizon 1: harness and baselines (TRK-16)

- [ ] Harness complete (section 3 above).
- [ ] Baselines from [14 §3.3](14-horizon-1-spec.md#33-baselines) running in the same harness, same backbone, same epochs, same replay budget.
- [ ] Each baseline passes the reproduction check against a published reference number; reference and our number recorded.
- [ ] The two static controls (pre-allocated and width-matched) and the two plasticity baselines (continual backprop, shrink-and-perturb) are *not* Month 1 work. They need the grown runs' final architectures and run in Month 3.

### Month 2 of Horizon 1: growth (TRK-13)

- [ ] `GrowableNetwork.grow(stage) -> GrowthEvent`, appending residual blocks with a zero-initialized last BatchNorm to a per-stage `nn.ModuleList` ([14 §4.1](14-horizon-1-spec.md#41-growth-unit-zero-initialized-residual-block)).
- [ ] `GrowthController.check(step, signals) -> GrowthDecision` as a pure function ([14 §4.2](14-horizon-1-spec.md#42-trigger)).
- [ ] Optimizer handling on growth ([14 §4.3](14-horizon-1-spec.md#43-optimizer-handling)) and online EWC that tolerates new parameters ([14 §4.4](14-horizon-1-spec.md#44-online-ewc-with-growth)).
- [ ] Uncertainty service: MC dropout + temperature scaling, with ECE, misclassification AUROC and AURC ([14 §4.5](14-horizon-1-spec.md#45-uncertainty-service)).
- [ ] Checkpoints with an architecture manifest; rebuild, then load.
- [ ] All required tests ([14 §5](14-horizon-1-spec.md#5-implementation-constraints)) passing in CI:
  1. Output identical before and after `grow()`, in train and eval mode.
  2. New parameters are in the optimizer.
  3. EWC penalty is 0 for parameters created after the last snapshot.
  4. Controller never grows in task 1, during warm-up or cooldown, past the cap or per-stage limit, or on rising or cross-boundary loss.
  5. Same seed gives same metrics on CPU.
  6. Checkpoint round-trip with a grown architecture.
  7. Trigger schedule feasible on every benchmark (already in `tests/test_config.py`).
  8. Calibration metrics match hand-computed values; temperature scaling never changes the predicted class.
- [ ] Pilot on Split-MNIST and on a few CIFAR tasks. Feed the measured per-run cost back into [14 §6](14-horizon-1-spec.md#6-compute-budget), and the measured variance of paired differences into the gate seed count.
- [ ] Controller thresholds tuned on the tuning stream, then frozen with the configs and tagged `prereg/gate-1`, with the advisor's sign-off ([PLAN.md §3](PLAN.md#decision-rule-all-gates)).

### Month 3 of Horizon 1: sweep and gate (TRK-12)

- [ ] Hyperparameters tuned on the tuning stream only.
- [ ] Grown runs first. Then the pre-allocated and width-matched controls from their final architectures, then the plasticity baselines, random-time and boundary controls.
- [ ] Full sweep: the main table on both tracks and both scenarios, plus the ablations (including the replay-budget sweep) in [14 §3.4](14-horizon-1-spec.md#34-ablations-track-a-class-incremental-split-cifar-100) (Track A, class-incremental).
- [ ] Gate 1 report written from the template in [14 §8](14-horizon-1-spec.md#8-gate-1-report-template), with the decision against the criteria in [PLAN.md §3.1](PLAN.md#31-gate-1-end-of-month-3), the Track B scope decision, and the outcome against the priors in [PLAN.md §6.2](PLAN.md#62-risk-register). The report is published, Go or No-Go.

---

## 5. Conventions

- **Seeds.** Every run takes an explicit `seed`. Evaluation seeds and the tuning-stream seed are defined in [14 §3.1](14-horizon-1-spec.md#31-streams); use them, do not invent others.
- **Configs are committed.** A result whose config is not in the repository does not count.
- **One experiment = one config + one JSONL log.** The log is the record; plots and tables are generated from it.
- **Never tune on evaluation seeds.** All hyperparameter decisions use the tuning stream's validation split. Evaluation test sets inform no decision.
- **Safe checkpoint loading.** Use `torch.load(..., weights_only=True)` or safetensors. Rebuild the architecture from the manifest before `load_state_dict`.
- **Never override `nn.Module.add_module`.** PyTorch calls it internally. Growth goes through `grow()`.
- **Plain PyTorch training loop in H1.** Single GPU; no Lightning, no DDP. Both assume a fixed parameter set.
- **Vectorized tensor ops only.** No per-neuron Python objects ([PLAN.md §5](PLAN.md#5-decisions-on-record)).
- **Tooling.** Ruff for lint and format, mypy, pytest, GitHub Actions. Library code lives under `src/ebrain/`.

---

## 6. Common Pitfalls

- **Per-neuron Python loops.** Anything that iterates over neurons in Python is off-plan and orders of magnitude too slow. Use batched tensor ops.
- **Growth that is not function-preserving.** If outputs change at the moment of growth, the new block's last BatchNorm γ/β was not zero-initialized, or the block is not purely residual. Test 1 catches this.
- **Forgetting to add new parameters to the optimizer.** New blocks then never train, and growth looks useless. Test 2 catches this.
- **Tuning on test.** Choosing any hyperparameter, checkpoint or early-stopping point using evaluation-seed test data invalidates the gate.
- **Unfair baselines.** Comparing against a static network that does not have the same final capacity, or that uses a different replay budget, backbone, augmentation or epoch count, proves nothing. The key control is the **pre-allocated** network: the same final architecture from task 1, with the *same* CL method.
- **Reading a failed test as equality.** "Random timing did not lose" is not "random timing matches". Equality needs the equivalence test in [PLAN.md §3](PLAN.md#decision-rule-all-gates).
- **Unequal memory.** Every store the learner trains from counts toward the byte budget ([PLAN.md §1](PLAN.md#operating-regime)). A method that keeps extra logits, anchors or exemplars must be compared at equal total memory or flagged.
- **Counting head expansion as growth.** The expanding classifier head is excluded from growth and from the parameter budget.

---

## 7. Hardware

See [PLAN.md §4](PLAN.md#4-budget-and-team) for H1 hardware, compute and storage. Unit tests and Split-MNIST smoke runs should work on CPU.

---

## 8. Learning Resources

Continual learning, foundations and scenarios:

- Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", PNAS 2017 (EWC).
- Schwarz et al., "Progress & Compress: A scalable framework for continual learning", ICML 2018 (online EWC).
- Zenke, Poole and Ganguli, "Continual Learning Through Synaptic Intelligence", ICML 2017 (SI).
- Lopez-Paz and Ranzato, "Gradient Episodic Memory for Continual Learning", NeurIPS 2017 (GEM; defines BWT and FWT).
- van de Ven and Tolias, "Three scenarios for continual learning", arXiv 2019; extended as van de Ven, Tuytelaars and Tolias, "Three types of incremental learning", Nature Machine Intelligence 2022.

Replay:

- Chaudhry et al., "Efficient Lifelong Learning with A-GEM", ICLR 2019.
- Chaudhry et al., "On Tiny Episodic Memories in Continual Learning", arXiv 2019 (ER baseline).
- Buzzega et al., "Dark Experience for General Continual Learning: a Strong, Simple Baseline", NeurIPS 2020 (DER++).

Growth and parameter isolation:

- Rusu et al., "Progressive Neural Networks", arXiv 2016.
- Mallya and Lazebnik, "PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning", CVPR 2018.
- Yoon et al., "Lifelong Learning with Dynamically Expandable Networks", ICLR 2018 (DEN).
- Chen, Goodfellow and Shlens, "Net2Net: Accelerating Learning via Knowledge Transfer", ICLR 2016 (function-preserving growth).
- Houlsby et al., "Parameter-Efficient Transfer Learning for NLP", ICML 2019 (residual adapters).

Libraries:

- Avalanche (ContinualAI): https://github.com/ContinualAI/avalanche
- Mammoth (AImageLab): https://github.com/aimagelab/mammoth
