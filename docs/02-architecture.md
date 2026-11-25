# E-Brain Technical Architecture

## Overview

E-Brain is a modular, dynamically growing neural system trained through developmental stages. §0 gives the design foundations, §1 the scheduled components as short cards, and §2 maps the earlier v1 components onto them. The v1 component designs are in [Appendix B](appendix-b-architecture-v1-unscheduled.md), unscheduled.

## 0. Design Foundations

E-Brain takes *principles* from biology and *implements* them with standard, vectorized deep learning:

- activity-dependent plasticity;
- structural growth and pruning;
- developmental curricula;
- homeostasis and sparsity.

The earlier per-neuron design (dendritic branches, leaky membrane, STDP, homeostasis, neuromodulation, neurogenesis) is now an off-critical-path research branch. See [Appendix A](appendix-a-bio-inspired-neuron-research-branch.md).

Schedule, gates and budgets are in [PLAN.md](PLAN.md). The Horizon 1 growth mechanism is specified in [14-horizon-1-spec.md](14-horizon-1-spec.md).

---

## 0.1 Implementation Strategy: Biological Inspiration vs. Vectorized Deep Learning

A key architectural distinction in E-Brain is between **biological principles** and **computational implementation**:

```
┌────────────────────────────────────────────────────────────────────────┐
│ Biological Principles (what inspires us):                              │
│ - Structural growth when capacity is short (neurogenesis)              │
│ - Consolidation of old knowledge (synaptic consolidation, replay)      │
│ - Developmental progression & sensory grounding                        │
│ - (Research branch only: STDP, spiking, homeostasis; see Appendix A)   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Realized via
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Vectorized Deep Learning Engine (how we compute):                      │
│ - Native PyTorch GPU tensor operations (batch-parallelized)            │
│ - `GrowableNetwork.grow()`: function-preserving residual blocks        │
│ - Online EWC + experience replay, consolidated at task boundaries      │
│ - Pretrained, frozen scaffolds with trainable adapters                 │
└────────────────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Implementation Rule**: E-Brain does **not** simulate millions of individual Python neuron objects with scalar loop updates. All biological mechanisms—including leaky integration, synaptic plasticity, and modular expansion—are implemented as **vectorized, batched PyTorch tensor operations** to leverage CUDA hardware acceleration.

---

## 0.2 Dual-Process Cognitive Architecture (System 1 vs. System 2)

To resolve the tension between human cognitive constraints (bounded working memory) and machine computational advantages (unlimited vector recall, parallel rollouts), E-Brain adopts a **Dual-Process Cognitive Architecture**:

```
                               ┌──────────────────────────────────────────────┐
                               │           E-Brain Dual-Process Engine        │
                               └──────────────────────┬───────────────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       ▼                                                             ▼
┌─────────────────────────────────────────────┐               ┌─────────────────────────────────────────────┐
│      System 1: Intuitive Substrate          │               │      System 2: Deliberative Controller      │
│      (Machine-Scale Substrate)              │               │      (Human-Inspired Cognitive Control)     │
├─────────────────────────────────────────────┤               ├─────────────────────────────────────────────┤
│ • Batched forward pass & perception         │               │ • Bounded active-goal slots (hyperparameter)│
│ • Parallel thought rollouts & search        │ ◄───────────► │ • Serial deliberation & attention           │
│ • Vector database retrieval (exact recall)  │   Feedback    │ • Goal priority management & hypothesis eval│
│ • Read-only tools (retrieval, lookup)       │   & Control   │ • Verification; gated side-effecting tools  │
└─────────────────────────────────────────────┘               └─────────────────────────────────────────────┘
```

1. **System 1 (Intuitive Substrate)** handles:
   - sensory encoding;
   - automatic pattern recognition;
   - associative memory retrieval;
   - parallel rollouts and search;
   - **read-only** tools.

   Retrieval capacity at inference is bounded by storage, not by a cognitive limit. It is separate from **training memory**: everything the learner trains from counts toward a fixed byte budget, and baselines get the same retrieval ([PLAN.md §1](PLAN.md#operating-regime)).
2. **System 2 (Deliberative Controller)** holds a **bounded number of active goal slots**. A slot is one open entry in the deliberation search frontier, ranked by an external verifier; the slot count is the frontier width, default 4, ablated over {2, 4, 7} at equal compute. See [PLAN.md §2](PLAN.md#2-terminology) and [16 §3](16-horizon-3-spec.md#3-system-2-slots-as-the-search-frontier). The bound is a *control* choice. It serializes high-level reasoning, prevents combinatorial drift and tracks current goals. It is not a claim about the hardware's capacity.
3. **Side-effecting tools** (subprocess, file writes, external APIs) are called *only* through System 2. Every call goes through the tool governance layer: sandbox, capability-scoped registry, approval gates and audit log (TRK-17, [13 §6.1](13-design-review-2026-11.md)). No side-effecting tool code is written before that spec is approved.

---

## 0.3 Foundation-Scaffolded Hybrid Architecture

Rather than training a multi-billion parameter model from scratch on minimal data—which would fail to produce emergent linguistic and abstract reasoning—E-Brain employs a **foundation-scaffolded approach**:

- **Perceptual and linguistic cortices.** Lightweight open-weight foundation models provide robust, pre-aligned visual and linguistic representations. They are frozen, or tuned only through adapters. Examples: CLIP ViT-B/32 and MobileNetV4, plus a SmolLM2-family LM whose size is fixed by a pilot ([PLAN.md §5](PLAN.md#5-decisions-on-record)). **Language is never trained from scratch.**
- **Dynamic associative cortex and growth core.** E-Brain's novel architecture sits on top of these representations. It develops:
  - Dynamic capacity addition (`GrowableNetwork.grow()`) with function-preserving residual blocks ([14 §4](14-horizon-1-spec.md#4-growth-mechanism)).
  - Hierarchical concept graphs with role-filler binding (compositional symbol grounding).
  - Continual learning and synaptic consolidation. The target is *bounded* forgetting, measured as BWT ([PLAN.md §3](PLAN.md#3-horizons-gates-and-benchmarks)); it is not "zero".
  - Embodied agency through a learned world model (H2). Theory of Mind is deferred beyond H3.

> **The earlier §1–§7** (v1 component designs) are now [Appendix B](appendix-b-architecture-v1-unscheduled.md), unscheduled. They predate the November 2026 review and contain known defects; §1 and §2 below replace them.

---

## 0.4 Architectural Alignment: The Three Horizons

Components are staged by horizon. Months, gates and benchmarks are in [PLAN.md §3](PLAN.md#3-horizons-gates-and-benchmarks).

1. **Horizon 1: Growth + Continual Learning**
   - `GrowableNetwork` growing by residual blocks with a zero-initialized last BatchNorm.
   - `GrowthController`: a plateau ∧ (underfit ∨ replay interference) trigger, with warm-up, cooldown and a parameter cap.
   - Online EWC + experience replay.
   - Full spec: [14-horizon-1-spec.md](14-horizon-1-spec.md).
2. **Horizon 2: Grounded Concepts + Agency** ([15](15-horizon-2-spec.md))
   - BabyAI agent: learned symbolic-grid encoder, frozen CLIP text encoder with FiLM, recurrent policy trained by imitation.
   - One action-conditioned world model; counterfactual agency detector built on it.
   - Concept graph with role-filler binding and a frozen held-out composition split.
   - Growth tested again on a level sequence of rising difficulty, with a learning-progress curriculum.
3. **Horizon 3: Deliberation, Tools, Adapters**
   - Propose → verify → revise deliberation over the LM scaffold, with System 2 slots as a bounded search frontier ([16](16-horizon-3-spec.md)).
   - Governed tool sandbox (TRK-17).
   - Same-lineage adapter clones + distillation. Cross-architecture concept transfer is a research question.

---

## 1. Component Cards

Each scheduled component has one card: what it is for, its horizon, its inputs and outputs, what trains it, how it is evaluated, and where it is specified. A component without a card is not scheduled. Cards for H2 and H3 are filled in to the level of their horizon spec before that horizon's coding starts (TRK-25).

### C1. Growable core and growth controller

| | |
|---|---|
| **Horizon** | H1; extended in H2 (encoder and pre-head blocks) and H3 (adapter allocation) |
| **Inputs → outputs** | Images or features (H1), symbolic observations (H2) → features and logits. The controller takes per-check signals → a grow / no-grow decision. |
| **Trained by** | The task loss plus C2's penalties. The controller has no learned parameters; its thresholds are tuned on the tuning stream. |
| **Evaluated by** | Gate 1; Gate 2 development; the function-preservation and controller unit tests |
| **Spec** | [14 §4](14-horizon-1-spec.md#4-growth-mechanism), [15 §3](15-horizon-2-spec.md#3-level-sequence-and-continual-learning), [16 §5](16-horizon-3-spec.md#5-evaluation-hygiene) |

### C2. Continual-learning protection and training-memory store

| | |
|---|---|
| **Horizon** | H1 onward |
| **Inputs → outputs** | Parameters and stored exemplars or trajectories → a penalty term and replay batches |
| **Trained by** | Not trained. Online EWC (H1), Progress & Compress with CLEAR-style replay (H2). Consolidation happens at task boundaries, and never rescales weights. |
| **Budget** | Everything here counts toward the training-memory budget in bytes ([PLAN.md §1](PLAN.md#operating-regime)) |
| **Evaluated by** | BWT, forgetting, memory in bytes |
| **Spec** | [14 §3.1, §4.4](14-horizon-1-spec.md#44-online-ewc-with-growth), [15 §2](15-horizon-2-spec.md#2-interface) |

### C3. Uncertainty service

| | |
|---|---|
| **Horizon** | H1 onward |
| **Inputs → outputs** | A model and an input → calibrated probabilities, predictive entropy, a scalar confidence |
| **Trained by** | Temperature scaling on validation data; MC dropout (H1); the world-model ensemble (H2) |
| **Evaluated by** | ECE, misclassification AUROC and AURC, after every task and every growth event |
| **Consumers** | Growth trigger variant, concept novelty, curiosity, self-assessment, H3 halting and escalation |
| **Spec** | [14 §4.5](14-horizon-1-spec.md#45-uncertainty-service), [15 §6](15-horizon-2-spec.md#6-uncertainty-consumers) |

### C4. Pretrained scaffolds and adapters

| | |
|---|---|
| **Horizon** | H1 Track B (CLIP image encoder); H2 (CLIP image and text encoders, FiLM); H3 (SmolLM2 + LoRA adapters) |
| **Inputs → outputs** | Raw images or text → frozen features; adapters add learned deltas |
| **Trained by** | Never the scaffold itself. Only adapter and FiLM parameters train, and only they count as E-Brain's learned parameters. |
| **Evaluated by** | Track B against nearest-class-mean; H3 clones (Gate 3 specialization) |
| **Spec** | [02 §0.3](#03-foundation-scaffolded-hybrid-architecture), [14 §2](14-horizon-1-spec.md#2-tracks), [16 §6](16-horizon-3-spec.md#6-adapters-and-clones) |

### C5. World model

| | |
|---|---|
| **Horizon** | H2 |
| **Inputs → outputs** | Encoded observation and action → predicted next latent, reward and termination; ensemble disagreement |
| **Trained by** | Per-cell observation reconstruction, reward and termination prediction, latent KL; ensemble heads regress the next latent |
| **Evaluated by** | k-step prediction error; agency AUROC through C8; ensemble calibration through C3 |
| **Spec** | [15 §5](15-horizon-2-spec.md#5-world-model-agency-and-intrinsic-reward) |

### C6. Policy with recurrent state

| | |
|---|---|
| **Horizon** | H2 |
| **Inputs → outputs** | Encoded observation, FiLM-conditioned on the mission → action distribution and value. The recurrent state is H2's working memory. |
| **Trained by** | Behavioural cloning from the BabyAI bot; optional PPO fine-tune |
| **Evaluated by** | Success-rate matrix over the level sequence (ACC and BWT in success rate); held-out composition success |
| **Spec** | [15 §2–§4](15-horizon-2-spec.md#2-interface) |

### C7. Concept graph v2

| | |
|---|---|
| **Horizon** | H2 |
| **Inputs → outputs** | Encoder features and attribute labels → concept nodes with role-filler bindings and grounding IDs into C2's store |
| **Trained by** | Supervised attribute detectors; the binding scheme and creation and merge rules are TRK-19 |
| **Evaluated by** | Gate 2 composition |
| **Spec** | [15 §7](15-horizon-2-spec.md#7-concept-graph), TRK-19 |

### C8. Agency detector and capability calibration

| | |
|---|---|
| **Horizon** | H2 |
| **Inputs → outputs** | C5's predicted outcomes under the taken and alternative actions → an agency score. The mission and state → the agent's predicted chance of its own success. |
| **Trained by** | Nothing new for agency, which is read off C5. Self-prediction trains on the agent's own outcomes. |
| **Evaluated by** | Gate 2 agency (AUROC against simulator counterfactuals); ECE and Brier score of self-predictions |
| **Spec** | [15 §5–§6](15-horizon-2-spec.md#5-world-model-agency-and-intrinsic-reward) |

### C9. Reward and alignment constraint

| | |
|---|---|
| **Horizon** | H2 (environment reward + one disagreement bonus); H3 (verifier value) |
| **Inputs → outputs** | Environment reward and C5's disagreement → the training reward. Proposed actions → allowed or blocked. |
| **Trained by** | Not trained. Alignment is a constraint that fails closed, never a weighted reward term. |
| **Evaluated by** | Zero constraint violations on a test suite, defined by TRK-18 |
| **Spec** | TRK-18; [13 §6.2](13-design-review-2026-11.md#62-reward-design-0241034521) |

### C10. Deliberation loop with a System 2 frontier

| | |
|---|---|
| **Horizon** | H3 |
| **Inputs → outputs** | A problem → an answer and its verified trace. The frontier holds at most W open entries (the slots). |
| **Trained by** | Adapters on the scaffold; a value model trained on externally verified outcomes |
| **Evaluated by** | Gate 3 deliberation; GSM-Symbolic; the frontier-width ablation at equal compute |
| **Spec** | [16 §3–§5](16-horizon-3-spec.md#3-system-2-slots-as-the-search-frontier), TRK-20 |

### C11. Tool gateway

| | |
|---|---|
| **Horizon** | H3 |
| **Inputs → outputs** | A tool request from System 2 (side-effecting) or System 1 (read-only) → the result and an audit record |
| **Trained by** | Not trained. Sandbox, capability-scoped registry, approval gates and audit log. |
| **Evaluated by** | Gate 3 tool safety (red-team suite) |
| **Spec** | TRK-17; [13 §6.1](13-design-review-2026-11.md#61-tool-governance-add-a-spec-before-any-h3-tool-code) |

**Infrastructure, not a component:** the CL harness (streams, metrics, JSONL logs), which replaces v1's performance monitor ([14 §3](14-horizon-1-spec.md#3-protocol)).

---

## 2. V1 Component Map

Every component of the v1 design ([Appendix B](appendix-b-architecture-v1-unscheduled.md)) and where it went. The reasoning is in [13 §11.3](13-design-review-2026-11.md#113-doc-02-17-component-triage-and-code-level-defects).

| v1 component | Verdict | Now |
|---|---|---|
| Dynamic core, growth controller | Keep | C1 |
| Performance monitor | Keep | CL harness |
| Vision encoder; language generator | Keep, as scaffolds | C4 |
| Text encoder ("BERT-style"), CNN+ViT hybrid | Replace | C4 |
| Working memory | Redesign | C6 (H2 recurrent state), C10 (H3 slots) |
| Multi-stage reasoning | Redesign, H3 outer loop only | C10 |
| Long-term memory | Redesign as a budgeted store with grounding IDs | C2, C7 |
| Uncertainty estimator | Redesign | C3 |
| Concept graph, concept-driven learning | Redesign | C7 |
| Entity tracking | Redesign as H2 object tracks or slots | C7 |
| Decision making | Redesign as a discrete policy with policy CL | C6 |
| Developmental reward system | Redesign | C9 |
| Self-identity | Redesign as agency + capability calibration | C8 |
| Audio encoder, motor control, Theory of Mind module, multi-agent coordinator, peer communication protocol | Defer beyond H3 | — |
| Meta-learning system | Defer beyond H3 (no gate needs it) | — |
| Unified embedding space | Cut (no gate needs it) | — |
| Concurrent thought system | Cut (duplicates the System 2 slots) | C10 |
| Timing system | Cut | — |
| Sensory-grounded thought system | Cut, replaced by the world model | C5 |
| Correction mechanism | Cut, merged into H3 revise, replay and the TRK-23 quarantine | C10, C2 |
| Person-perspective system | Cut (the scaffold handles deixis) | — |

---

See [PLAN.md](PLAN.md) for when each horizon runs, and [04-implementation-roadmap.md](04-implementation-roadmap.md) for the build order.

---
*Architecture Version: 2.0*
*Last Updated: November 25, 2026*
