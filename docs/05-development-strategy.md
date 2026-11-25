# Development Strategy

- **Status**: Active. Rewritten to match the canonical plan (TRK-14).
- **Last updated**: November 25, 2026
- **Replaces**: the earlier "implement everything first, then train" version of this document. The reasons for the rewrite are in [13-design-review-2026-11.md](13-design-review-2026-11.md) §3, §4, §5.4, §5.6 and §8.

> [!IMPORTANT]
> **Schedule, gates, budget and team live only in [PLAN.md](PLAN.md).** This document describes *how* each horizon is built, trained and evaluated. It does not state gate months, gate thresholds, GPU-hour budgets, costs or team size. Where it needs one of those numbers, it links to PLAN.md. If anything here seems to disagree with PLAN.md, PLAN.md wins and this document is the bug.

**Terminology** follows [PLAN.md §2](PLAN.md#2-terminology):

- **Horizon 1 / 2 / 3** are the only units of schedule.
- **Developmental Phase 0–5** are curriculum stages inside the model's training, not project steps. H1 covers Phase 0, H2 covers Phases 1–2, H3 covers Phases 3–4, and Phase 5 is beyond H3.
- Project work inside a horizon is described in **steps**, never "phases".

---

## 1. Principles

1. **Build and train together.** Every component is trained on a small experiment as soon as it exists. There is no long "implementation only" period. In H1, the first deliverable of the continual-learning harness is a set of reproduced, *trained* baselines ([14 §7](14-horizon-1-spec.md#7-month-by-month)).
2. **Vectorized PyTorch only.** The substrate is batched tensor operations on the GPU. No per-neuron Python objects, no Python loops over neurons, no spiking/STDP on the critical path ([PLAN §5](PLAN.md#5-decisions-on-record)). Spiking and STDP are a research branch: [Appendix A](appendix-a-bio-inspired-neuron-research-branch.md).
3. **Pretrained scaffolds for perception and language.** Language is never trained from scratch. It comes from a SmolLM2-family scaffold plus trainable adapters in H3 ([02 §0.3](02-architecture.md#03-foundation-scaffolded-hybrid-architecture)).
4. **Pre-register the evaluation.** Protocol, splits, metrics, baselines and the gate statistic are written down and frozen before the runs they judge. Held-out data is never used for any decision.
5. **Procedurally generated environments first.** They give unlimited data, exact ground truth, and controllable splits. Real-world datasets come in only where a benchmark requires them.
6. **Measure, don't claim.** Every capability is reported as a number on a named benchmark with a confidence interval, never as "zero forgetting" or "perfect recall" ([13 §8](13-design-review-2026-11.md#8-claims-to-reword)).

---

## 2. Development Workflow

Each horizon runs the same loop, many times, at increasing scale:

```
  implement  ──>  test  ──>  experiment  ──>  analyze
      ^                                          │
      └──────────────────────────────────────────┘
```

| Step | What it means | Done when |
|---|---|---|
| **Implement** | A small, reviewable change: one module, one CL method, one environment wrapper | Code is typed, linted (Ruff), and documented |
| **Test** | Unit tests for the invariants of that change (e.g. "output identical before and after `grow()`") | Tests pass in CI on CPU |
| **Experiment** | A small run that exercises the change end to end: a Split-MNIST pass in H1, a few hundred thousand env steps in H2, a GSM8K dev slice in H3 | Run is logged per §2.1 |
| **Analyze** | Compare to the previous run and to the relevant baseline. Decide: keep, revise, or drop | A one-paragraph note in the experiment log |

**Every component lands with unit tests and a small experiment in the same pull request.** A component that has never been trained is not "done".

Scale up only after the small experiment works: smoke test, then pilot (a few tasks or a short env budget), then the full sweep with all seeds. Gate sweeps are run once, from frozen configs.

### 2.1 Experiment log discipline

- **One JSONL event log per run.** Task boundaries, growth events, evaluation metrics, and trigger signal values, each with step and wall-clock time ([14 §5](14-horizon-1-spec.md#5-implementation-constraints)). A W&B mirror is optional; the JSONL file is the record.
- **Everything needed to rerun is captured.** The fully resolved Hydra config, the seed(s), the git commit SHA and a dirty-tree flag, library versions, dataset identifiers and checksums, and the hardware.
- **Configs are committed.** Every run is launched from a config under `configs/`. Gate runs use configs that are committed and tagged before the sweep starts.
- **An append-only experiment index** (one JSONL row per run: id, config path, commit, seeds, headline metrics, one-line conclusion) is committed to the repo. Raw logs and checkpoints stay outside git.
- **Seeds are explicit.** Each seed fixes initialization, data order and (in H1) the class-to-task assignment. Gate runs use the seed count from [PLAN §3](PLAN.md#3-horizons-gates-and-benchmarks).
- **Tuning is separate from evaluation.** Hyperparameters are tuned on a separate held-out stream or split, never on the one being evaluated.
- **Reproduction before trust.** A baseline implementation, ours or a library's, must reproduce a published reference number on that reference's own configuration before its results count ([14 §3.3](14-horizon-1-spec.md#33-baselines)).

### 2.2 Checkpoints and data pipeline

- **Checkpoint at every task boundary and every growth event**, not just at the end. A failed run resumes from the last good boundary.
- **Grown models need an architecture manifest.** Store the list of added blocks per stage next to the `state_dict`. To load, rebuild the architecture from the manifest, then call `load_state_dict`.
- **Safe loading only.** Use `torch.load(path, weights_only=True)` or safetensors. Never unpickle an untrusted file.
- **Deterministic data.** Splits are generated from a seed and saved as index files, so every method sees exactly the same stream.
- **Precompute frozen features once.** When a backbone is frozen (H1 Track B, H2 perception), cache its features to disk and train on the cache.
- **Datasets are versioned by checksum**, and the license of each source is recorded next to it (§4).

---

## 3. Per-Horizon Strategy

### 3.1 Horizon 1: Growth + Continual Learning (Developmental Phase 0)

**The full how-to is [14-horizon-1-spec.md](14-horizon-1-spec.md).** This section is a summary; doc 14 is authoritative for every detail.

**Implemented**
- A continual-learning harness: streams, metrics (ACC, BWT, FWT, forgetting), JSONL logging, a Hydra config tree (`model/`, `growth/`, `cl_method/`, `optimizer/`, `benchmark/`, `seed`).
- Baselines in that harness, per [14 §3.3](14-horizon-1-spec.md#33-baselines): fine-tune, joint, online EWC, SI, ER, A-GEM, DER++, ER-ACE, DER at equal memory, PackNet and Progressive Nets (task-incremental only). Also the two static controls: **pre-allocated** (the grown final architecture from task 1) and **width-matched**, and two plasticity baselines on the pre-allocated architecture: **continual backprop** and **shrink-and-perturb**. They test whether growth is just a plasticity injection.
- `GrowableNetwork` growing by whole residual blocks with a zero-initialized last BatchNorm, so growth is exactly function-preserving. The API is `grow(stage) -> GrowthEvent`.
- `GrowthController.check(step, signals) -> GrowthDecision`, a pure function. The trigger is plateau ∧ (underfit ∨ replay interference). Timing is in fractions of the task, with warm-up, cooldown, no growth in task 1, and a parameter cap checked as params + unit ≤ cap.
- Online EWC that tolerates new parameters (zero Fisher until the next snapshot), and reservoir replay.
- An uncertainty service (MC dropout + temperature scaling) shared by every consumer of uncertainty, evaluated with ECE, misclassification AUROC and AURC ([14 §4.5](14-horizon-1-spec.md#45-uncertainty-service)).

**Trained**
- From the start, beginning with the baselines. Two tracks: **A, from scratch** (a reduced ResNet-18, gated) and **B, frozen CLIP features** (reported, compared against nearest-class-mean on the same features).

**Evaluation highlights**
- Split-MNIST (5 tasks) is the smoke test and the CI-sized run.
- **Class-incremental Split-CIFAR-100 (10 tasks) is gated.** Task-incremental results are reported alongside but do not count.
- Equal replay budget for every method, under the bounded-memory regime in [PLAN §1](PLAN.md#operating-regime). Memory is reported in bytes (parameters + CL state + exemplars), and a replay-budget sweep shows how the result depends on it. The primary control is the pre-allocated final architecture; results are paired by seed.
- Ablations include **random-time growth** and **boundary growth**, which test whether the controller's timing matters at all.
- Gate conditions and the No-Go adaptation path: [PLAN §3.1](PLAN.md#31-gate-1-end-of-month-3) and [§6](PLAN.md#6-adaptation-triggers-and-risk-register). Track B also decides whether H2–H3 grow on top of frozen encoders.

**Pitfalls**
- Plain PyTorch, single GPU. Lightning and DDP assume a fixed parameter set, and growth changes it mid-run.
- After growth, register the new parameters with `optimizer.add_param_group(...)` and leave existing state untouched.
- **Never override `nn.Module.add_module`**; PyTorch uses it internally. Append blocks to a per-stage `nn.ModuleList` so existing keys never change.
- No width growth in H1. No RL in H1.
- A minimal sketch of the Track B growth unit, a residual MLP block on the frozen-feature trunk. Track A instead appends whole BasicBlocks whose last BatchNorm γ/β is zero-initialized. This is for orientation only; doc 14 §4.1 is the spec:

```python
import torch
from torch import nn


class ResidualAdapter(nn.Module):
    """y = x + W_up(GELU(W_down(LN(x)))), with W_up zero-initialized so y == x at insertion."""

    def __init__(self, dim: int, reduction: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, dim // reduction)
        self.up = nn.Linear(dim // reduction, dim)
        nn.init.kaiming_normal_(self.down.weight, nonlinearity="relu")
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(nn.functional.gelu(self.down(self.norm(x))))
```

### 3.2 Horizon 2: Grounded Concepts + Agency (Developmental Phases 1–2)

H2 puts the H1 substrate inside an interactive environment where difficulty rises, and tests growth again there. It starts only after Gate 1 passes, and after the H2 spec ([15](15-horizon-2-spec.md)) is complete and the Gate 2 split is frozen (TRK-19, TRK-27, [PLAN §7](PLAN.md#7-next-steps)). The interface, level sequence, curriculum and world model are fixed in doc 15; this section covers how to build them.

**Environments**
- **MiniGrid and BabyAI** through the maintained `gymnasium` and `minigrid` packages. The old `gym` / `gym_minigrid` packages and the 4-tuple `step` API are not used.
- **An in-house agency environment** built on MiniGrid, with **independent movers**: objects or agents that move on their own, so some outcomes are not caused by the agent.
- Environments are procedurally generated. Seeds define levels; there is no dataset to download.

```python
import gymnasium as gym
import minigrid  # noqa: F401  (registers the MiniGrid and BabyAI environments)

env = gym.make("BabyAI-GoToObj-v0")
obs, info = env.reset(seed=1234)
done = False
while not done:
    action = env.action_space.sample()  # replaced by the E-Brain policy
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

For throughput, run many environments in parallel with `gymnasium.vector.AsyncVectorEnv` on CPU cores and batch the policy forward pass on the GPU.

**Implemented and trained, in order**
1. **Perception.** A learned CNN over the symbolic grid (primary, where growth applies), and the frozen CLIP image encoder over RGB renders (secondary). Missions go through the frozen CLIP text encoder and FiLM ([15 §2](15-horizon-2-spec.md#2-interface)). Cache frozen features where possible.
2. **Supervised level-0 attribute detectors.** MiniGrid exposes ground truth for every cell (object type, colour, state, position). Use it to train attribute heads with a supervised loss. This sidesteps the open problem of discovering atomic concepts without labels ([13 §5.3](13-design-review-2026-11.md)).
3. **Role-filler concept graph.** Compositions bind attributes to roles (e.g. colour-of-object-1, type-of-object-1) with VSA/HRR circular convolution or slot-attention object slots with attribute heads. Sum-of-embeddings composition is not allowed: it cannot distinguish "red key, blue ball" from "blue key, red ball". New concepts need minimum support and a PMI threshold, near-duplicates are merged, confidence is capped (Beta posterior), and each concept stores the IDs of its grounding examples so it can be re-embedded after growth or consolidation.
4. **Counterfactual agency detector.** "Caused by my action" means the outcome would have differed under a different action from the same state. Ground-truth labels come from counterfactual rollouts: clone the environment, including its RNG state, apply each alternative action, and compare outcomes. The detector is trained on observed transitions and evaluated against those labels in the environment with independent movers. A correlation between actions and outcomes, or an `agency_score` that ignores the action taken, is not acceptable.
5. **The world model.** One action-conditioned latent dynamics model with a 5-member ensemble ([15 §5](15-horizon-2-spec.md#5-world-model-agency-and-intrinsic-reward)). Agency scores, the intrinsic reward and self-prediction all read from it; there are no separate imagery, timing or curiosity modules.
6. **The agent.** **The E-Brain network is the policy and value network**: a recurrent core with policy and value heads on the `GrowableNetwork` encoder, with the concept graph's outputs as part of the state. It is trained first by **imitation from the BabyAI bot**, with an optional **PPO** fine-tune. PPO is written in-repo (single-file, CleanRL style) or taken from a library that accepts an arbitrary `nn.Module`. Library default policies such as an `MlpPolicy` are not used, because then E-Brain's network is never the policy. Continual learning across levels uses Progress & Compress and CLEAR-style replay. Growth reuses the H1 rules (zero-initialized residual blocks, `add_param_group`, cooldown, cap), with the policy signals in [15 §3](15-horizon-2-spec.md#3-level-sequence-and-continual-learning).
7. **The curriculum.** A learning-progress level sampler, with fixed-order, shuffled and joint arms reported against it ([15 §4](15-horizon-2-spec.md#4-curriculum)). No fixed exit thresholds.

```python
import copy


def action_caused_outcome(env, action, alternatives, outcome) -> bool:
    """Ground-truth agency label from counterfactual rollouts.

    `env` is the unwrapped MiniGrid env (render_mode=None). Deep copies carry the
    grid, the agent state and the RNG, so independent movers behave identically
    across branches. `outcome` maps an env to a hashable outcome of interest.
    """
    factual = copy.deepcopy(env)
    factual.step(action)
    observed = outcome(factual)
    for alt in alternatives:
        branch = copy.deepcopy(env)
        branch.step(alt)
        if outcome(branch) != observed:
            return True
    return False
```

**Evaluation protocol highlights**
- **A systematic held-out composition split** over (colour × object × task), defined and frozen *before* any H2 training. Held-out combinations never appear in training levels, demonstrations or tuning.
- The key controls are a parameter-matched agent **without** the concept graph, trained with the same algorithm and environment-step budget, and the **pre-allocated static agent** on the level sequence, which tests growth where difficulty rises.
- Agency is scored as AUROC against counterfactual labels. Retention on the H2 level sequence is reported with BWT in success rate, as in H1.
- Gate thresholds: [PLAN §3.2](PLAN.md#32-gate-2-end-of-month-8).

**Sample budgets.** Plan in **millions of environment steps** per level family, not in episodes. BabyAI levels are known to need on the order of millions of RL steps, or many bot demonstrations, before they are solved reliably. Measure steps-to-threshold in a pilot, and size the sweep from that measurement against the budget in [PLAN §4](PLAN.md#4-budget-and-team). Bot demonstrations are generated locally, count toward the training-memory budget, and are given in the same amount to every control agent.

**Consolidation.** Replay and EWC snapshots are triggered by training signals (task boundary, replay-buffer fill, drift on a probe set, or every N steps) through a Consolidation Scheduler. There is no wall-clock "day/night" schedule (TRK-21).

**Pitfalls**
- Leakage of held-out compositions through mission strings, demonstrations, or hyperparameter selection.
- Agency labels that are confounded by movers whose RNG differs between branches.
- Concept explosion from raw co-occurrence; concept embeddings that drift silently after growth.
- Reward hacking of intrinsic bonuses. Curiosity is the world model's ensemble disagreement, and only actions inside the simulator earn it (TRK-18).

### 3.3 Horizon 3: Deliberation, Tools, Adapters (Developmental Phases 3–4)

H3 adds language, deliberation, governed tools and specialization on top of a pretrained language model. It starts after Gate 2. The tool governance spec and the reward redesign (TRK-17, TRK-18) must be approved before H3 tool work begins.

**Scaffold and adapters**
- **Base model: a SmolLM2-family instruct model, kept frozen.** Its size is fixed by a pilot before Month 9 ([PLAN §5](PLAN.md#5-decisions-on-record)): the smallest model whose single-pass program-aided GSM8K dev accuracy is 20–60%. Starting at 360M may well end at 1.7B. E-Brain's trainable parts are LoRA or other adapters, plus projection heads that connect H2 perception and concepts to the LM.
- This is the only route to language (Developmental Phase 3). There is no next-word training from scratch on WikiText or any other corpus, and no perplexity target.

**Deliberation: propose → verify → revise**
- An outer loop over the scaffold: the model proposes an answer or a program, an **external** verifier checks it (execution result, unit check, consistency check), and the model revises, within a fixed step and token budget.
- **System 2 slots are the search frontier**: at most W open sub-goals or partial programs, ranked by verifier value. W ∈ {2, 4, 7} is ablated at equal compute ([16 §3](16-horizon-3-spec.md#3-system-2-slots-as-the-search-frontier)).
- **GSM8K is program-aided.** The model writes a short Python program, the program is executed, and its result is verified. Program execution is a tool, so it runs only inside the sandbox defined by TRK-17: no network, no host filesystem, resource limits and a timeout.
- The gated comparison is against **compute-matched self-consistency**: a majority vote over program-aided samples, with the same base model, adapters, token budget and execution budget. Single-pass results are reported too, and the single-pass baseline also executes its program, so the gain measures revising rather than running code. GSM-Symbolic is reported next to GSM8K. Every capability claim is a delta over the scaffold alone ([16 §5](16-horizon-3-spec.md#5-evaluation-hygiene)). Tune on a dev split carved from GSM8K train; touch GSM8K test only for the final report. Check for contamination of the test set in the adapter training data.
- **ARC-AGI-1 public eval** is tracked and reported, not gated.
- Inside the network, adaptive depth uses weight-tied iteration with learned halting (ACT/PonderNet), not hardwired stage roles (TRK-20).

**Theory of Mind (reported, not gated)**
- Evaluate on **held-out templates and held-out story structures**, in the style of ToMi and Hi-ToM splits: more agents, deeper belief nesting, and distractor events than anything seen in training.
- Training items, if any, are generated from templates disjoint from the evaluation templates. Also report on at least one external ToM benchmark that was never used for training or tuning.

**Tools**
- **No tool code before TRK-17 is approved** ([PLAN §5](PLAN.md#5-decisions-on-record)). Then: sandboxed execution, a capability-scoped registry with risk tiers, human approval for side-effecting calls, a secrets broker, an append-only audit log, bounded retries and parallelism, and exploration only against mocks or recorded replays ([13 §6.1](13-design-review-2026-11.md#61-tool-governance-add-a-spec-before-any-h3-tool-code)).
- Side-effecting tools belong to System 2 behind governance. Read-only retrieval may stay in System 1.
- A red-team suite for sandbox escapes and prompt injection is built alongside the sandbox, not after it.

**Specialization: same-lineage adapter clones**
- A clone is the same frozen base plus a per-domain adapter. Knowledge moves between clones through adapter merging (task arithmetic, TIES) and output-level distillation on a shared probe set.
- Each clone is compared against a full fine-tune of the same base on its domain, and its base-domain accuracy is measured with the adapter active. The growth controller may allocate adapters per domain; that is compared with one adapter per domain ([16 §5](16-horizon-3-spec.md#5-evaluation-hygiene)). Cross-architecture "concept package" transfer is a research question, not part of the H3 deliverable.

**Alignment** is a hard constraint that fails closed, never a weighted reward term. Corrections keep full weight at every stage (TRK-18).

Gate thresholds: [PLAN §3.3](PLAN.md#33-gate-3-end-of-month-14).

**Pitfalls**
- Claiming deliberation gains that come from a bigger compute budget, a different model, or different adapters.
- Expecting high absolute GSM8K scores from a small model; the claim is the *relative*, compute-matched gain of the loop.
- Letting tool output, web content or peer messages become instructions or training targets without filtering.

### 3.4 Beyond Horizon 3 (unscheduled)

Developmental Phase 5 (expertise), multi-model ecosystems, cross-architecture concept transfer, and spiking/STDP substrates are **unscheduled** until Gate 3 passes ([PLAN §1](PLAN.md#1-scope-in-one-paragraph)). They are re-planned then, in PLAN.md. Spiking/STDP work may continue as an off-critical-path research branch under [Appendix A](appendix-a-bio-inspired-neuron-research-branch.md); nothing in H1–H3 depends on it.

---

## 4. Data Requirements

Sizes are approximate. Storage and compute budgets are in [PLAN §4](PLAN.md#4-budget-and-team). Verify each license against the source before use and record it next to the dataset checksum.

| Horizon | Dataset / environment | Approximate size | Role | License note |
|---|---|---|---|---|
| H1 | MNIST (Split-MNIST, 5 tasks) | 70k images, ~12 MB | Smoke test | No formal license; widely redistributed for research |
| H1 | CIFAR-100 (Split-CIFAR-100, 10 tasks) | 60k images, ~170 MB | Gated benchmark | No formal license; free for research use |
| H1 | CLIP ViT-B/32 image encoder | ~350 MB weights; cached features a few hundred MB | Track B frozen features | MIT (OpenAI CLIP); check the specific checkpoint if using OpenCLIP weights |
| H2 | MiniGrid / BabyAI | Procedural; no download | Agent training and held-out compositions | Apache-2.0 (`minigrid` package) |
| H2 | Agency environment with independent movers | Procedural; no download | Agency detection with counterfactual labels | Ours (Apache-2.0) |
| H2 | BabyAI bot demonstrations (optional) | Generated locally; size set by the H2 spec | Imitation pretraining for agent and control alike | Generated from Apache-2.0 code |
| H3 | SmolLM2-360M-Instruct or SmolLM2-1.7B-Instruct (pilot decides) | ~0.7 GB or ~3.4 GB (bf16) | Frozen language scaffold | Apache-2.0 |
| H3 | GSM8K | ~7.5k train, ~1.3k test problems | Deliberation (program-aided) | MIT |
| H3 | ARC-AGI-1 | 400 training + 400 public eval tasks | Tracked, not gated | Apache-2.0 |
| H3 | ToMi / Hi-ToM-style generators | Generated; thousands of stories | Held-out ToM evaluation | Research releases; check each, some are non-commercial |
| H3 | Domain-clone suite | Chosen in the H3 spec | Adapter specialization | Permissive sources only; record per source |
| H3 | Sandbox red-team suite | Ours; hundreds of cases | Tool safety | Ours (Apache-2.0) |

Not used: WikiText-103 or any corpus for training language from scratch; hand-written ToM items that are both trained and evaluated on; moving-square toy data as an exit criterion.

---

## 5. Roles and Skills

These are responsibilities, not headcount. Team size is in [PLAN §4](PLAN.md#4-budget-and-team). One person can hold several roles.

| Horizon | Role | Responsibilities | Skills |
|---|---|---|---|
| All | **Evaluation and infrastructure** | Harness, CI, experiment index, seeds and statistics, reproduction checks, gate reports | PyTorch, pytest, statistics (CIs, Welch tests), MLOps |
| All | **Advisor / reviewer** | Reviews pre-registered protocols and gate reports; challenges novelty claims | Continual learning and RL research experience |
| H1 | **Continual-learning engineer** | `GrowableNetwork`, growth controller, online EWC + replay, baselines | PyTorch internals (modules, optimizers, state dicts), CL literature |
| H2 | **RL and environments engineer** | PPO with the E-Brain network as policy, vectorized rollouts, MiniGrid custom environments, counterfactual labeling | RL (PPO), gymnasium/minigrid, CPU/GPU throughput |
| H2 | **Representation engineer** | Attribute detectors, role-filler binding, concept creation and drift policy | VSA/HRR or slot attention, representation learning |
| H3 | **LM adaptation engineer** | SmolLM2 adapters (LoRA), deliberation loop and verifiers, adapter clones and merging | Hugging Face Transformers and PEFT, LM evaluation |
| H3 | **Security and safety engineer** | TRK-17 sandbox and registry, audit log, red-team suite, alignment constraints | Container/microVM isolation (gVisor, Firecracker, nsjail), threat modeling |

---

## 6. What Not to Do

- **Per-neuron Python objects or loops over neurons.** Use batched tensor operations. Spiking/STDP stays in Appendix A.
- **Training language from scratch.** Language comes from the pretrained scaffold plus adapters.
- **Wall-clock circadian schedules.** Consolidation is triggered by training signals, not by the time of day.
- **Evaluating on training items.** Held-out templates, structures, compositions and task streams are fixed before training and never used for tuning.
- **Tool code before governance.** Nothing executes untrusted code or calls external services until TRK-17 is approved.
- **Unbounded training memory.** Everything the learner trains from counts toward a fixed byte budget ([PLAN §1](PLAN.md#operating-regime)).
- **Deployment or learning from real users before Gate 3** and the TRK-17/TRK-23 safety review ([PLAN §5](PLAN.md#5-decisions-on-record), D-16).
- Also: overriding `nn.Module` methods, using deprecated `gym`/`gym_minigrid` APIs, using a library default policy instead of the E-Brain network, and comparing against unmatched or untuned baselines.

---

## 7. Related Documents

- [PLAN.md](PLAN.md): canonical schedule, gates, benchmarks, budget, team and terminology
- [DECISIONS.md](DECISIONS.md): context and alternatives for each decision on record
- [14-horizon-1-spec.md](14-horizon-1-spec.md): Horizon 1 protocol, growth mechanism and implementation constraints
- [15-horizon-2-spec.md](15-horizon-2-spec.md), [16-horizon-3-spec.md](16-horizon-3-spec.md): Horizon 2 and 3 spec drafts
- [13-design-review-2026-11.md](13-design-review-2026-11.md): the review behind this rewrite
- [02-architecture.md](02-architecture.md): architecture, vectorized implementation rule, System 1 / System 2, foundation scaffold, component cards
- [03-developmental-phases.md](03-developmental-phases.md): Developmental Phases 0–5
- [Appendix A: bio-inspired neuron research branch](appendix-a-bio-inspired-neuron-research-branch.md): spiking/STDP, off the critical path
