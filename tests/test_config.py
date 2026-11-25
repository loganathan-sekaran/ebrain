"""The Hydra config tree must compose, keep each value in one place, and give a feasible trigger
schedule on every benchmark (docs/14-horizon-1-spec.md §4.2, required test 7)."""

import math
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
BENCHMARKS = sorted(p.stem for p in (CONFIG_DIR / "benchmark").glob("*.yaml"))


def _compose(overrides: list[str] | None = None) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name="config", overrides=overrides or [])


def test_default_config_composes() -> None:
    cfg = _compose()
    assert cfg.benchmark.name == "split_cifar100"
    assert cfg.benchmark.num_tasks == 10
    assert cfg.scenario == "class_incremental"
    assert cfg.model.growth_unit.zero_init_last_bn is True
    assert cfg.optimizer.schedule == "constant_per_task"


def test_growth_cap_matches_gate_1() -> None:
    assert _compose().growth.param_budget_multiplier == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("benchmark", "expected"), [("split_cifar100", 2000), ("split_mnist", 500)]
)
def test_replay_size_has_one_source(benchmark: str, expected: int) -> None:
    cfg = _compose([f"benchmark={benchmark}"])
    assert cfg.cl_method.replay.buffer_size == expected
    assert cfg.cl_method.replay.batch_size == cfg.benchmark.batch_size


def _achievable_events(cfg: DictConfig) -> int:
    """Greedy count of growth events the schedule allows in one task, ignoring the signals."""
    b, g = cfg.benchmark, cfg.growth
    train = math.floor(b.train_images_per_task_min * (1 - b.val_fraction))
    steps = (train // b.batch_size) * b.epochs_per_task
    interval = max(1, math.floor(steps * g.check_every_frac))
    n_checks = steps // interval
    # Check k (1-based) needs W earlier checks in this task and must be past the warm-up.
    first = max(g.plateau_window + 1, math.ceil(steps * g.warmup_frac / interval))
    gap = math.ceil(steps * g.cooldown_frac / interval)
    events, k = 0, first
    while k <= n_checks and events < g.max_events_per_task:
        events += 1
        k += gap
    return events


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_trigger_schedule_is_feasible(benchmark: str) -> None:
    cfg = _compose([f"benchmark={benchmark}"])
    assert _achievable_events(cfg) >= cfg.growth.max_events_per_task
