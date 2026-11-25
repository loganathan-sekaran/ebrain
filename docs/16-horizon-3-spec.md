# Horizon 3 Specification: Deliberation, Tools, Adapters

- **Status**: Skeleton draft (November 25, 2026). It fixes what a System 2 slot is, the evaluation hygiene and the adapter-allocation test, from recommendations R1, R8 and R9 of [13 §11.2](13-design-review-2026-11.md#112-second-pass-design-review-recommendations-applied). The tool governance spec (TRK-17), the reward redesign (TRK-18) and the per-run budget are still open (§7).
- **Schedule, gates and budget**: [PLAN.md §3.3, §4](PLAN.md). Gate thresholds and totals live only there.
- **Depends on**: the Gate 2 report, and the Track B scope decision ([PLAN.md §3.1](PLAN.md#31-gate-1-end-of-month-3)).

---

## 1. What H3 Adds

H3 puts deliberation on top of a pretrained language-model scaffold, and asks whether a bounded propose → verify → revise loop beats spending the same compute on independent samples. It also adds governed tool use and same-lineage adapter clones.

---

## 2. Base-Model Pilot

The rule is decision D-04 in [PLAN.md §5](PLAN.md#5-decisions-on-record). Procedure:

1. Carve a GSM8K dev split from the **train** set (500 problems, fixed seed). The test set is never used for this.
2. For each SmolLM2 instruct size (135M, 360M, 1.7B), run single-pass program-aided inference: the model writes a Python program, and the sandboxed executor (TRK-17) runs it.
3. Pick the smallest size whose accuracy falls in the range D-04 sets. Record all three numbers in the H3 spec.

---

## 3. System 2 Slots as the Search Frontier

A **slot** is one open entry in the deliberation frontier: a sub-goal or a partial program. The bounded slot count of [PLAN.md §2](PLAN.md#2-terminology) is the frontier width.

- **Propose.** The LM scaffold with its adapters extends the highest-value frontier entries with candidate next steps or whole programs.
- **Verify.** An **external** verifier scores each candidate: program execution in the sandbox, type and unit checks, consistency between independent derivations, and later a learned value model trained on verified outcomes. The model's own unprompted judgment of its answer is not a verifier: intrinsic self-correction without external feedback does not reliably improve reasoning (Huang et al., "Large Language Models Cannot Self-Correct Reasoning Yet", ICLR 2024).
- **Revise.** Keep the top W entries by verifier value, where W is the slot count, and drop the rest.
- **Halt.** When the uncertainty service's calibrated confidence in the best entry passes a threshold set on the dev split, or the compute budget runs out. Low confidence at the budget limit escalates instead of answering.
- **Ablation.** W ∈ {2, 4, 7} at **equal compute**: the same tokens and program executions per problem. Width trades breadth against depth.

---

## 4. Baselines

| Baseline | Definition |
|---|---|
| Single-pass, program-aided | One program, executed. It also runs Python, so the deliberation delta measures *revising*, not *running code*. |
| Compute-matched self-consistency | Majority vote over program-aided samples with the same base model, adapters, token budget and execution budget. The Gate 3 comparison. |
| Frontier width ablation | §3 |

---

## 5. Evaluation Hygiene

- **Contamination.** Report **GSM-Symbolic** (Mirzadeh et al. 2024), which regenerates GSM8K problems from templates, next to GSM8K test, with the same comparison. Check adapter training data for n-gram overlap with both test sets, and report the result.
- **The executor is a tool.** It runs in the TRK-17 sandbox, and the red-team suite covers it.
- **Theory of Mind** is reported as the gain over the scaffold alone on the same probes, never as an absolute score that the scaffold already reaches.
- **ARC-AGI-1** public eval is tracked, not gated.
- **Adapter allocation (developmental test in H3).** The growth controller decides when a domain gets a new adapter and when it reuses an existing one, using the signals of [14 §4.2](14-horizon-1-spec.md#42-trigger) on the domain's validation split. This is compared with one adapter per domain at equal total adapter parameters. Reported with Gate 3.

---

## 6. Adapters and Clones

- Same-lineage LoRA adapters on the chosen base, plus distillation ([PLAN.md §5](PLAN.md#5-decisions-on-record), D-17).
- The reference for each clone is a full fine-tune of the same base on the same domain data.
- Base-domain accuracy is measured with the adapter active.

---

## 7. Open Before H3 Starts

- Tool governance spec (TRK-17), including the executor sandbox and the red-team suite.
- Reward redesign (TRK-18).
- The base-model pilot (§2).
- The verifier's value model: training data, and how it is kept out of the test sets.
- The per-run compute breakdown within the H3 cap in [PLAN.md §4](PLAN.md#4-budget-and-team).
- The domain-clone suite: which domains, which data, which metrics.
- The `prereg/gate-3` tag ([PLAN.md §3](PLAN.md#decision-rule-all-gates)).
