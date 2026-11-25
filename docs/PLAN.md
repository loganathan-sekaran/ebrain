# E-Brain Canonical Plan

> **This is the single source of truth** for timelines, gate thresholds and decision rules, benchmarks, budget totals and team.
>
> If another document disagrees with this one, this one wins. That document is a bug: fix it, or point it here.
>
> Other docs explain *how* and *why*, and link here for *when* and *how much*. Horizon specs may hold protocol and implementation parameters, the per-run breakdown of their budget, and a month-by-month breakdown *within* their horizon. They do not restate gate thresholds, horizon month ranges or totals. `tests/test_single_source.py` enforces this in CI (§8).

- **Status**: Active
- **Owner**: Project lead
- **Last updated**: November 25, 2026 (R1–R13 of the second-pass review applied; see [13 §11.2](13-design-review-2026-11.md#112-second-pass-design-review-recommendations-applied))
- **Changes since doc 12**: See [13-design-review-2026-11.md](13-design-review-2026-11.md) (TRK-11)
- **Context for each decision**: [DECISIONS.md](DECISIONS.md)

---

## 1. Scope in One Paragraph

E-Brain tests one central hypothesis, in three stages:

> A network that grows capacity when it detects pressure, and protects old knowledge as it grows, learns a stream of tasks better than the same network with that capacity allocated from the start.

1. **Horizon 1** tests that claim on standard continual-learning benchmarks.
2. **Horizon 2** grounds it in an interactive environment with compositional concepts and agency, and tests it again where task difficulty actually rises.
3. **Horizon 3** adds deliberation, governed tool use and adapter-based specialization on top of pretrained models.

Anything beyond Horizon 3 is unscheduled until Gate 3 passes. That includes expert-level domains, multi-model ecosystems and spiking/STDP substrates.

### Operating regime

The hypothesis is only meaningful under **bounded memory and bounded compute**. With unlimited exact memory, replaying everything would solve forgetting and make growth moot. So:

- **Bounded training memory.** Every store the learner *trains from* counts toward one memory budget, in bytes: replay exemplars, CL state (Fisher, anchors, stored logits), episodic memory, stored corrections and any distilled "insights". Each horizon spec states the budget. Gate comparisons are at equal budget.
- **Bounded compute per step.** Every method gets the same number of optimizer steps (H1), environment steps (H2), or tokens and program executions (H3). FLOPs per step are reported, since growth changes them.
- **Retrieval at inference is separate.** System 1 may retrieve from a store that the learner does not train from. Its size is reported, and every baseline gets the same retrieval.

---

## 2. Terminology

| Term | Meaning | Do not use it for |
|---|---|---|
| **Horizon 1 / 2 / 3** | Project stages with go/no-go gates. They are the only unit of *schedule*. | — |
| **Month N** | Calendar months counted from **Month 1 = the month TRK-15 (repo hygiene) merges**. The target is December 2026. | Absolute dates in other docs |
| **Developmental Phase 0–5** | Curriculum stages inside the model's training (see [03](03-developmental-phases.md)). They are *not* schedule units. Progress through them is driven by a learning-progress sampler, not by fixed exit thresholds ([15 §4](15-horizon-2-spec.md#4-curriculum)). | Project steps ("Phase 1: implementation") |
| **System 1** | Parallel, unbounded substrate: perception, retrieval, batched search, read-only tools | Side-effecting actions |
| **System 2** | Deliberative controller with a **bounded number of active goal slots**. A slot is one open entry in the deliberation search frontier: a sub-goal or partial program, ranked by an external verifier's value. The slot count is the frontier width: default 4, ablated over {2, 4, 7} at equal compute ([16 §3](16-horizon-3-spec.md#3-system-2-slots-as-the-search-frontier)). | Claims of "unlimited working memory" |
| **Training memory** | Everything the learner trains from, in bytes (§1) | Inference-time retrieval stores |

### Developmental phases mapped to horizons

| Phase | Name | Horizon | Note |
|---|---|---|---|
| 0 | Initialization | H1 | Substrate only: `GrowableNetwork` + continual learning (CL) |
| 1 | Sensory | H2 | MiniGrid observations, supervised attribute detectors |
| 2 | Sensorimotor / agency | H2 | World model, counterfactual agency detection |
| 3 | Language | H3 | Through the pretrained LM scaffold. Not trained from scratch. |
| 4 | Abstract reasoning | H3 | Deliberation loop + verifiers + tools |
| 5 | Expertise | Beyond H3 | Unscheduled |

---

## 3. Horizons, Gates and Benchmarks

| Horizon | Months | Builds | Benchmarks | Gate |
|---|---|---|---|---|
| **H1: Growth + Continual Learning** | 1–3 | `GrowableNetwork` (function-preserving residual-block growth), growth controller, online EWC + replay, uncertainty service, CL harness and baselines | Class-incremental **Split-CIFAR-100 (10 tasks)**; Split-MNIST (5 tasks) as a smoke test; task-incremental results reported alongside | Gate 1 (§3.1) |
| **H2: Grounded Concepts + Agency** | 4–8 | BabyAI agent on the H1 substrate ([15](15-horizon-2-spec.md)): world model, recurrent policy, role-binding concept graph, counterfactual agency detector, learning-progress curriculum | MiniGrid/BabyAI with a **systematic held-out (colour × object × task) split**; the BabyAI level sequence; an agency environment with independent movers | Gate 2 (§3.2) |
| **H3: Deliberation, Tools, Adapters** | 9–14 | LM scaffold (SmolLM2 family, size fixed by a pilot; §5) + adapters, propose → verify → revise loop over a bounded frontier, governed tool sandbox, same-lineage adapter clones ([16](16-horizon-3-spec.md)) | GSM8K test (program-aided), GSM-Symbolic (reported), ARC-AGI-1 public eval (tracked, not gated), a domain-clone suite, a sandbox red-team suite | Gate 3 (§3.3) |
| **Beyond H3** | unscheduled | Expertise, cross-architecture concept transfer, ecosystem | — | Re-planned after Gate 3 |

### Decision rule (all gates)

Seeds pair runs across methods: the same seed gives the same class order (or level order) and initialization. So every comparison uses **per-seed paired differences** d = A − B (H3: per-problem pairs).

- **A beats B by m**: the lower bound of the 95% CI of mean(d) is **≥ m**.
- **A is non-inferior to B within m**: that lower bound is **≥ −m**.
- **A matches B within m**: the whole 95% CI lies inside [−m, m] (TOST equivalence). A failed "beats" test never counts as "matches".

A gate passes only if **every** condition passes. This is an intersection–union test, so the Go decision needs no multiplicity correction.

**Seeds.**
- H1 gate comparisons use 10 seeds. That number is raised, never lowered, if the Month 2 pilot's paired-difference variance requires it.
- H2 gates use ≥5 seeds, with the same raise-only pilot rule.
- H3 uses paired per-problem tests over the full test set.

**Tuning.** Hyperparameters are tuned on a separate tuning stream. In H1 that is the same data in a different class order, with decisions made only on validation slices. Tuning never touches evaluation-seed test data.

**Pre-registration.** Before a gate sweep starts, the spec, configs and analysis script are frozen in a git tag `prereg/gate-N`, with the advisor's sign-off recorded in the tag message (§8). A result from a config that differs from the tag does not count toward the gate.

### 3.1 Gate 1 (end of Month 3)

These metrics are defined in [14-horizon-1-spec.md §3.2](14-horizon-1-spec.md#32-metrics). The controls are defined in [§3.3](14-horizon-1-spec.md#33-baselines) and [§3.4](14-horizon-1-spec.md#34-ablations-track-a-class-incremental-split-cifar-100).

**Go** if *all* of the following hold on Track A, class-incremental Split-CIFAR-100 (10 tasks), with an equal replay budget:
1. **Accuracy.** Triggered growth beats the **pre-allocated static control** by **≥ 2 points ACC**. That control is the same final architecture present from task 1, trained with the same CL method.
2. **Retention.** BWT is **non-inferior** to the pre-allocated control **within 1 point**.
3. **Size.** Final backbone parameters are **≤ 2× the initial** count in every run.
4. **Timing.** Triggered growth beats **random-time growth** (same number of events, at feasible times) by **> 0 points ACC**.
5. **Schedule.** Triggered growth is **non-inferior** to **boundary growth** (same number of events) **within 1 point ACC**.

**No-Go** if any condition fails. There is no undefined middle band. On No-Go, apply the [§6.1 adaptation triggers](#61-adaptation-triggers).

**Reported with Gate 1, not gated:**
- **Track B scope decision.** Growth on frozen CLIP features keeps its place in H2–H3 *on top of frozen encoders* only if it beats **both** nearest-class-mean and the pre-allocated trunk by **> 0 points ACC** (class-incremental, paired). Otherwise H2–H3 grow only where features are learned, such as the H2 symbolic-observation encoder.
- **Plasticity baselines.** Continual backprop and shrink-and-perturb on the pre-allocated architecture ([14 §3.3](14-horizon-1-spec.md#33-baselines)). If either matches triggered growth within 1 point ACC, the paper says growth works as a plasticity injection, and H2 keeps the cheaper method as a baseline.

### 3.2 Gate 2 (end of Month 8)

**Go** if all hold:
1. **Composition.** Mean success on **held-out compositions** is **≥ 80%**, *and* E-Brain's held-out error is **at most half** the error of a parameter-matched agent without the concept graph. That agent uses the same algorithm and environment-step budget. "At most half" means the upper 95% bound of the paired error ratio is ≤ 0.5.
2. **Agency.** Agency detection reaches a mean **AUROC ≥ 0.90**, with a lower 95% bound ≥ 0.85, for "outcome caused by my action". This is measured in an environment with independent movers, with ground truth from counterfactual rollouts.
3. **Retention.** On the H2 level sequence ([15 §3](15-horizon-2-spec.md#3-level-sequence-and-continual-learning)), BWT in success rate is **non-inferior** to the matched control **within 2 points**.
4. **Development.** On the same level sequence, where difficulty rises, triggered growth beats the **pre-allocated static agent** (the grown agent's final architecture from the start, same algorithm, same environment steps) by **≥ 2 points** of final mean success.

**Reported with Gate 2, not gated:** the curriculum arms in [15 §4](15-horizon-2-spec.md#4-curriculum) (fixed difficulty order, learning-progress sampler, shuffled, joint), at equal environment steps.

### 3.3 Gate 3 (end of Month 14)

**Go** if all hold:
1. **Deliberation.** The propose → verify → revise loop beats **compute-matched self-consistency** by **≥ 2 points on GSM8K test**, using paired per-problem tests. Self-consistency here means a majority vote over program-aided samples, with the same base model, adapters, token budget and program-execution budget.
2. **Tool safety.** **0 sandbox escapes** in the red-team suite, and **100%** of side-effecting calls logged and approved, as defined in the tool governance spec (TRK-17). The GSM8K program executor is in scope.
3. **Specialization.** Each adapter clone reaches **≥ 90%** of a *full* fine-tune of the same base on its domain. Its base-domain accuracy, with the adapter active, is **non-inferior** to the frozen base **within 2 points**.

**Reported with Gate 3, not gated:** the same comparison as condition 1 on GSM-Symbolic, and controller-allocated domain adapters against one adapter per domain ([16 §5](16-horizon-3-spec.md#5-evaluation-hygiene)). If the condition 1 gain does not hold on GSM-Symbolic (lower bound ≤ 0), the Gate 3 report must say so next to the decision.

---

## 4. Budget and Team

| Horizon | Compute (GPU-hours) | Hardware | Storage | Cloud cost at $1–1.5 per GPU-hour |
|---|---|---|---|---|
| H1 | ≤ 400 | 1 × 24 GB consumer GPU | < 20 GB | $400–600 |
| H2 | ≤ 1,500 | 1–2 × 24 GB GPU + multi-core CPU for env rollouts | < 50 GB | $1,500–2,250 |
| H3 | ≤ 3,000 | 1–2 × 24–48 GB GPU | < 200 GB | $3,000–4,500 |
| **Total to Gate 3** | **≤ 4,900** | | **< 300 GB** | **$4,900–7,350** |

Owning a GPU makes H1 and most of H2 nearly free. Each horizon spec must break its budget down per run. A horizon that exceeds its GPU-hour cap by more than 50% triggers a review.

**Team.** At minimum, 1 full-time research engineer plus a part-time advisor. Ideally, 2–3 engineers with one focused on infrastructure and evaluation. The timeline above assumes the minimum team. The advisor signs off each pre-registration (§3).

---

## 5. Decisions on Record

Each row has a short record (context, alternatives, consequences, when to revisit) in [DECISIONS.md](DECISIONS.md).

| ID | Decision | Choice | Rationale / doc |
|---|---|---|---|
| D-01 | Compute substrate | Vectorized PyTorch tensor ops. No per-neuron Python objects. | [02 §0.1](02-architecture.md#01-implementation-strategy-biological-inspiration-vs-vectorized-deep-learning) |
| D-02 | Spiking / STDP | Research branch only. Off the critical path. | [Appendix A](appendix-a-bio-inspired-neuron-research-branch.md) |
| D-03 | Perception and language | Pretrained scaffold (CLIP ViT-B/32 or MobileNetV4; SmolLM2 family) + trainable adapters. Language is never trained from scratch. | [02 §0.3](02-architecture.md#03-foundation-scaffolded-hybrid-architecture) |
| D-04 | H3 base-model size | Fixed by a pilot before Month 9. It is the smallest SmolLM2 instruct model whose single-pass program-aided accuracy on a GSM8K dev split, carved from train, is 20–60%. Models around 360M score near the floor on GSM8K, so deliberation gains could not be measured. | [16 §2](16-horizon-3-spec.md#2-base-model-pilot) |
| D-05 | H1 pretrained features | Run **both** tracks: from scratch (gated), and frozen-CLIP (reported). Track B decides whether H2–H3 grow on top of frozen encoders (§3.1). | [14 §2](14-horizon-1-spec.md#2-tracks) |
| D-06 | Growth unit (H1) | Whole residual blocks with a zero-initialized last BatchNorm, so insertion is exactly function-preserving. Small bottleneck adapters would add too little capacity for the 2× cap to matter. | [14 §4.1](14-horizon-1-spec.md#41-growth-unit-zero-initialized-residual-block) |
| D-07 | Training loop (H1) | Plain PyTorch. No Lightning, because growth changes the parameter set mid-run. | [14 §5](14-horizon-1-spec.md#5-implementation-constraints) |
| D-08 | Statistics | Paired by seed; decision rule and pre-registration as in §3 | §3 |
| D-09 | Operating regime | Bounded training memory (bytes) and bounded compute per step; inference retrieval separate and shared with baselines | §1 |
| D-10 | Uncertainty | One calibrated uncertainty service (ensembles or MC dropout, plus temperature scaling), evaluated with ECE, misclassification AUROC and risk–coverage. Every consumer uses it. | [02 §1, C3](02-architecture.md#c3-uncertainty-service) |
| D-11 | H2 learner | Symbolic grid observations (RGB secondary), frozen CLIP text encoder + FiLM for missions, imitation from the BabyAI bot with an optional PPO fine-tune, Progress & Compress + CLEAR-style replay for continual learning | [15 §2](15-horizon-2-spec.md#2-interface) |
| D-12 | H2 world model | One learned, action-conditioned latent dynamics model with an ensemble. Its disagreement is the intrinsic reward; agency is scored from it. It replaces the imagery, timing and self-prediction modules. | [15 §5](15-horizon-2-spec.md#5-world-model-agency-and-intrinsic-reward) |
| D-13 | Curriculum | A learning-progress level sampler, not fixed phase-exit thresholds | [15 §4](15-horizon-2-spec.md#4-curriculum) |
| D-14 | Alignment | A hard constraint that fails closed. Never a weighted reward term. | [13 §6.2](13-design-review-2026-11.md#62-reward-design-0241034521) (TRK-18) |
| D-15 | Tools | No tool code until the governance spec (TRK-17) is approved. Program execution for GSM8K counts as a tool and runs in that sandbox. | [13 §6.1](13-design-review-2026-11.md#61-tool-governance-add-a-spec-before-any-h3-tool-code) |
| D-16 | Deployment | No user-facing deployment and no learning from real users before Gate 3 passes **and** the TRK-17 and TRK-23 safety review is approved | [13 §11.2](13-design-review-2026-11.md#112-second-pass-design-review-recommendations-applied) |
| D-17 | Cloning | Same-lineage adapter clones + distillation. Cross-architecture concept transfer is research. | [13 §5.5](13-design-review-2026-11.md#55-cloning-and-knowledge-transfer-doc-07) |
| D-18 | Python / tooling | Python ≥3.12, Ruff, mypy, pytest, GitHub Actions | [pyproject.toml](../pyproject.toml) |
| D-19 | License | Apache-2.0 | [LICENSE](../LICENSE) |

---

## 6. Adaptation Triggers and Risk Register

### 6.1 Adaptation triggers

- **Gate 1 fails on condition 4 or 5** (random timing does as well, or boundary growth does better): drop the controller claim. Continue with the simplest schedule that works, for example growing at task boundaries, and reframe the contribution as an empirical study of growth at equal capacity.
- **Gate 1 fails on condition 1 or 2**: try at most two alternatives within one extra month: mixture-of-experts routing, or blocks per task with a task-inference head. Then re-gate or stop.
- **Track B shows no benefit** (§3.1): H2–H3 grow only where features are learned. This is a scope change, not a No-Go.
- **Gate 2 fails on composition**: before abandoning the concept graph, try a different binding scheme (VSA ↔ slot attention).
- **Gate 2 fails on development (condition 4)** while conditions 1–3 pass: drop growth from H3 and continue with a static substrate. The growth thesis is then reported as holding only in H1.
- **Fixed difficulty order does not beat shuffled** (§3.2, reported): drop the "developmental curriculum" claim and keep the learning-progress sampler as an engineering choice.
- **Budget overrun >50%** in a horizon: cut benchmarks before cutting seeds.

### 6.2 Risk register

The probabilities are initial estimates from the November 2026 review, not measurements. The project lead and advisor replace them with their own at each pre-registration, and the gate report compares the outcome with them.

| Risk | Prior | Early signal | Response |
|---|---|---|---|
| Gate 1 condition 1 fails (growth ≈ pre-allocated) | ~0.5 | Month 2 pilot: paired ACC difference near 0 on 3 CIFAR tasks | §6.1, then stop or reframe |
| Gate 1 condition 4 fails (timing does not matter) | ~0.5 | Pilot: random-time runs within noise of triggered runs | Drop the controller claim (§6.1) |
| Gate 1 passes overall | ~0.2 | — | — |
| Gate 2 passes, given Gate 1 passed | ~0.35 | Month 6: held-out success and agency AUROC on the dev split | §6.1 |
| Gate 3 passes, given Gate 2 passed | ~0.3 | Base-model pilot: single-pass accuracy outside 20–60% | Change model size before Month 9 |
| A baseline fails its reproduction check | ~0.3 per baseline | Month 1 | Use a reference implementation (Avalanche or Mammoth), or drop the baseline and say so |
| Compute overrun >50% | ~0.2 | Month 2 pilot cost | Cut benchmarks before seeds |
| Single-person dependency (illness, leave) | ongoing | — | Everything in the repo; the docs-and-tests CI keeps the plan runnable by someone new |
| Legacy docs pull work off-plan | ongoing | Work items citing docs 06–11 or Appendix B | PLAN.md wins; the single-source lint (§8) |

**When to stop.** The growth thesis stops if Gate 1 fails again after the one-month adaptation, or if Gate 2 condition 4 fails after Gate 1 passed. Stopping means publishing the negative result (§8). Any continuation needs a new plan with its own gates.

---

## 7. Next Steps

| When | What | Tracker |
|---|---|---|
| Now | Repo hygiene, CI, test skeleton | TRK-15 |
| Month 1 | CL harness + the baselines in [14 §3.3](14-horizon-1-spec.md#33-baselines), except the static controls and the plasticity baselines that run on their architecture | TRK-16 |
| Month 2 | `GrowableNetwork` + growth controller per [14 §4](14-horizon-1-spec.md#4-growth-mechanism), uncertainty service. The pilot sets the gate seed count; thresholds are frozen; `prereg/gate-1` tag. | TRK-13 |
| Month 3 | Grown runs → static controls → plasticity and timing controls → full sweep → Gate 1 report, published | TRK-12, TRK-26 |
| Before Month 4 | H2 spec completed from [15](15-horizon-2-spec.md): concept graph v2, world model, Gate 2 split frozen | TRK-19, TRK-27 |
| Before H2 coding | Doc 02 component cards C5–C9 filled in to the H2 spec's level | TRK-25 |
| Before Month 9 | Tool governance spec, reward redesign, H3 base-model pilot, H3 spec completed from [16](16-horizon-3-spec.md) | TRK-17, TRK-18 |

---

## 8. Process

- **Decision records.** Every row of §5 has a record in [DECISIONS.md](DECISIONS.md). A change to a decision changes both, in the same commit.
- **Single-source lint.** `tests/test_single_source.py` fails CI if a maintained doc restates a gate threshold or a horizon month range. Legacy docs are exempt until they are rewritten or archived; the exempt list is in the test.
- **Archive.** When a doc's replacement lands, the old doc moves to [docs/archive/](archive/README.md) with a banner, and links to it are updated.
- **Pre-registration.** The `prereg/gate-N` tag (§3) is pushed before the first gate-sweep run.
- **Publish every gate report**, No-Go included, as a preprint or workshop paper, so each gate gets outside review.
