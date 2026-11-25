# Decision Records

- **Last updated**: November 25, 2026
- **What this is**: one short record per row of [PLAN.md §5](PLAN.md#5-decisions-on-record). PLAN.md holds the decision itself; this file holds the context, the alternatives that were rejected, the consequences, and what would make us revisit it. A change to a decision updates both files in the same commit.

Status values: **Accepted**, **Superseded by D-xx**, **Under review**.

---

### D-01. Compute substrate

- **Status**: Accepted (TRK-14, November 2026)
- **Context**: The earlier design used per-neuron Python objects with dendrites, membrane state and STDP. That is orders of magnitude too slow to train on any benchmark in the plan.
- **Alternatives**: per-neuron objects; a spiking framework (snnTorch, Norse) on the critical path.
- **Consequences**: every mechanism on the critical path is a batched tensor op. Biological ideas survive as principles: growth, consolidation, curricula, sparsity.
- **Revisit when**: a spiking substrate matches the vectorized one on Gate 1 at equal compute (research branch, Appendix A).

### D-02. Spiking / STDP

- **Status**: Accepted (TRK-14)
- **Context**: Spiking and STDP add simulation cost and training difficulty, and no gate needs them.
- **Alternatives**: surrogate-gradient spiking networks as the main substrate.
- **Consequences**: the material lives in [Appendix A](appendix-a-bio-inspired-neuron-research-branch.md) as a research branch with no schedule.
- **Revisit when**: after Gate 3, or if the research branch produces a result worth gating.

### D-03. Perception and language

- **Status**: Accepted
- **Context**: Training perception and language from scratch on developmental data would not produce usable representations within the budget.
- **Alternatives**: training from scratch; large proprietary models through an API.
- **Consequences**: small open-weight scaffolds, frozen or tuned through adapters. Only adapter deltas count as E-Brain's learned parameters.
- **Revisit when**: a horizon gate shows that a frozen scaffold blocks the claim being tested.

### D-04. H3 base-model size

- **Status**: Accepted (self-review, November 2026)
- **Context**: The smallest SmolLM2 models score near the floor on GSM8K, so a deliberation gain over them would be noise.
- **Alternatives**: fixing SmolLM2-360M in advance; a larger non-SmolLM2 model.
- **Consequences**: a pilot before H3 picks the smallest model whose single-pass program-aided accuracy sits in the measurable range ([16 §2](16-horizon-3-spec.md#2-base-model-pilot)). The H3 compute cap still applies.
- **Revisit when**: no SmolLM2 size falls in the range, or the chosen size breaks the H3 budget.

### D-05. H1 pretrained features

- **Status**: Accepted; scope rule added November 25, 2026 (R2)
- **Context**: H2 and H3 run on frozen encoders. A growth result from scratch alone would not show that growth helps on top of them.
- **Alternatives**: only from scratch; only frozen features.
- **Consequences**: Track A gates; Track B decides whether H2–H3 grow on top of frozen encoders, using the rule in [PLAN.md §3.1](PLAN.md#31-gate-1-end-of-month-3).
- **Revisit when**: the Gate 1 report.

### D-06. Growth unit (H1)

- **Status**: Accepted (self-review)
- **Context**: Bottleneck adapters added about 5% capacity in total, so the parameter cap never applied and capacity could not plausibly matter.
- **Alternatives**: bottleneck adapters; width growth (Net2Net-style); per-task columns.
- **Consequences**: whole residual blocks with a zero-initialized last BatchNorm. One stage-4 block is a large step, so the cap applies after two.
- **Revisit when**: growth helps but the step size looks too coarse; width growth is then a follow-up.

### D-07. Training loop (H1)

- **Status**: Accepted
- **Context**: Lightning and DDP assume a fixed parameter set, while growth adds parameter groups mid-run.
- **Alternatives**: Lightning with custom callbacks; DDP with re-wrapping.
- **Consequences**: a plain PyTorch loop on one GPU in H1.
- **Revisit when**: H2 needs multi-GPU rollouts.

### D-08. Statistics

- **Status**: Accepted (self-review); pre-registration added November 25, 2026 (R12)
- **Context**: Earlier drafts mixed non-overlapping CIs with Welch tests, ignored the seed pairing, and read a failed superiority test as "matches".
- **Alternatives**: Welch t-tests; Bayesian estimation.
- **Consequences**: paired lower-bound rule, TOST for equivalence, intersection–union across conditions, a `prereg/gate-N` tag before each sweep.
- **Revisit when**: a pilot shows paired differences far from normal; switch to a paired bootstrap CI then, decided before the sweep.

### D-09. Operating regime

- **Status**: Accepted (R3)
- **Context**: Several old docs promise exact, storage-bounded memory. With unlimited replay, forgetting is solved and growth has nothing to add, so the thesis only holds under a stated budget.
- **Alternatives**: leaving memory unbounded and reporting it.
- **Consequences**: one training-memory budget in bytes per horizon, equal across methods at each gate; equal steps per method; inference retrieval kept separate and shared with baselines. H1 also reports a replay-budget sweep ([14 §3.4](14-horizon-1-spec.md#34-ablations-track-a-class-incremental-split-cifar-100)).
- **Revisit when**: never silently; a horizon spec that needs more memory states the new budget for every method.

### D-10. Uncertainty

- **Status**: Accepted (R6)
- **Context**: Uncertainty drives at least six things in the old docs (a trigger variant, concept novelty, halting, escalation, curiosity, self-assessment), and none of them was calibrated or evaluated.
- **Alternatives**: one ad hoc estimator per consumer.
- **Consequences**: one service ([02 C3](02-architecture.md#c3-uncertainty-service)), evaluated after every task and every growth event.
- **Revisit when**: a consumer needs a quantity the service cannot provide, such as epistemic versus aleatoric separation.

### D-11. H2 learner

- **Status**: Accepted (R7); details in [15 §2](15-horizon-2-spec.md#2-interface)
- **Context**: Doc 14's signals (validation loss, train accuracy, task boundaries) do not exist as such for a policy, and training a mission encoder from scratch would contradict D-03.
- **Alternatives**: pure PPO from scratch; RGB-only observations; a GRU mission encoder trained from scratch.
- **Consequences**: symbolic observations first (mirroring Track A), RGB second (mirroring Track B); frozen CLIP text encoder with FiLM; imitation from the BabyAI bot; policy continual learning with Progress & Compress and CLEAR-style replay.
- **Revisit when**: imitation fails to reach useful success on the easy levels.

### D-12. H2 world model

- **Status**: Accepted (R5); details in [15 §5](15-horizon-2-spec.md#5-world-model-agency-and-intrinsic-reward)
- **Context**: The old design had imagery decoders, a timing predictor, `SelfModel.predict` and hand-written curiosity terms, none with a loss or an evaluation.
- **Alternatives**: keeping separate modules; no world model (model-free only).
- **Consequences**: one action-conditioned latent dynamics model with an ensemble. Disagreement is the intrinsic reward; agency is scored from predicted outcomes under alternative actions.
- **Revisit when**: k-step prediction error stays too high for agency scoring to beat a model-free baseline.

### D-13. Curriculum

- **Status**: Accepted (R1)
- **Context**: Fixed exit thresholds (">80%") and calendar phases are arbitrary, and curricula often add little in standard settings, so the developmental claim needs a control.
- **Alternatives**: fixed thresholds; no curriculum.
- **Consequences**: a learning-progress level sampler, with fixed-order, shuffled and joint arms reported against it ([15 §4](15-horizon-2-spec.md#4-curriculum)).
- **Revisit when**: the Gate 2 report.

### D-14. Alignment

- **Status**: Accepted (TRK-18 pending)
- **Context**: A weighted alignment reward can be traded off against other rewards.
- **Alternatives**: reward shaping; RLHF-style preference reward.
- **Consequences**: alignment is a constraint that fails closed. The reward redesign is TRK-18.
- **Revisit when**: never weakened; refined by TRK-18.

### D-15. Tools

- **Status**: Accepted
- **Context**: The old tool code ran subprocesses and file writes with no sandbox.
- **Alternatives**: tools first, governance later.
- **Consequences**: no tool code before the TRK-17 spec. The GSM8K program executor is a tool and runs in the sandbox.
- **Revisit when**: TRK-17 is approved.

### D-16. Deployment

- **Status**: Accepted (R11)
- **Context**: The plan has no user-facing deployment, but older docs describe learning online from users, which needs trust, provenance and privacy work first.
- **Alternatives**: a limited beta during H3.
- **Consequences**: no deployment and no learning from real users before Gate 3 passes and the TRK-17 and TRK-23 safety review is approved.
- **Revisit when**: both conditions hold.

### D-17. Cloning

- **Status**: Accepted (TRK-22)
- **Context**: Transferring concepts between independently grown architectures is an open research problem.
- **Alternatives**: cross-architecture concept transfer as a scheduled feature.
- **Consequences**: same-lineage adapter clones plus distillation in H3; cross-architecture transfer is research.
- **Revisit when**: after Gate 3.

### D-18. Python and tooling

- **Status**: Accepted (TRK-15)
- **Context**: A new codebase can start on a current Python that every planned dependency supports, and one tool for linting and formatting keeps CI simple.
- **Alternatives**: Black + isort + flake8; Python 3.10.
- **Consequences**: Ruff, mypy strict, pytest, GitHub Actions on Python 3.12 and 3.13. [pyproject.toml](../pyproject.toml) is the source of truth.
- **Revisit when**: a dependency drops support for 3.12.

### D-19. License

- **Status**: Accepted
- **Context**: The project means to publish code and results.
- **Alternatives**: MIT; GPL-3.0.
- **Consequences**: Apache-2.0, which includes a patent grant.
- **Revisit when**: never, without the copyright holders' agreement.
