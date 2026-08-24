"""Benchmark script for `trace_rays` comparing `use_image_method=True` vs `False`
using pre-computed ECOS certified ground truth dataset.

Usage:
    uv run --group benchmarks python benchmarks/trace_rays.py
"""

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from fpt_jax import trace_rays


@dataclass(frozen=True)
class SolverConfig:
    name: str
    use_image_method: bool
    num_iters_linesearch: int
    line_style: str
    marker: str
    color: str


class ScenarioResult(TypedDict):
    ref_time_ms: float | None
    ref_label: str
    xs: dict[str, list[float]]
    ys: dict[str, list[float]]


def _mean_error(got: jax.Array, expected: jax.Array) -> float:
    diff = jnp.linalg.norm(got - expected, axis=-1)
    err = jnp.mean(diff, axis=-1)
    finite = jnp.isfinite(err)
    safe_err = jnp.where(finite, err, 1e2)
    count = jnp.sum(finite)
    mean_val = jnp.where(count > 0, jnp.sum(safe_err) / count, 1e2)
    val = float(mean_val)
    return max(val, 1e-7) if np.isfinite(val) else 1e2


def _bench_with_timeit(fn: Callable[[], jax.Array], *, repeats: int) -> float:
    times_ms = []
    _ = fn().block_until_ready()  # Warm-up
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = fn().block_until_ready()
        times_ms.append(1e3 * (time.perf_counter() - t0))
    return float(min(times_ms))


def _run_trace_rays(
    tx: jax.Array,
    rx: jax.Array,
    object_origins: jax.Array,
    object_vectors: jax.Array,
    interaction_types: jax.Array,
    *,
    use_image_method: bool,
    num_iters: int,
    num_iters_linesearch: int,
) -> jax.Array:
    y = trace_rays(
        tx,
        rx,
        object_origins,
        object_vectors,
        interaction_types=interaction_types,
        num_iters=num_iters,
        num_iters_linesearch=num_iters_linesearch,
        use_image_method=use_image_method,
    )
    return y.block_until_ready()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark `trace_rays` solvers against ECOS dataset")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("benchmarks/data/benchmark_dataset.npz"),
        help="Path to pre-computed ECOS dataset.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=128,
        help="Batch size to evaluate.",
    )
    parser.add_argument(
        "--ns",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="List of number of interactions (n) to benchmark.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 20, 50, 100],
        help="List of iteration counts for solvers.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeats for timing benchmarks.",
    )
    parser.add_argument(
        "--linesearch-iters",
        type=int,
        default=1,
        help="Number of linesearch iterations per solver step.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["A", "B", "C", "D"],
        default=["A", "B", "C", "D"],
        help="Scenarios to run.",
    )
    return parser.parse_args()


def run_benchmark(
    *,
    sc_name: str,
    p_reflect: float,
    num_dims: int,
    ns: list[int],
    iters_grid: tuple[int, ...],
    batch: int,
    repeats: int,
    solver_cfgs: list[SolverConfig],
    dataset_npz: dict[str, np.ndarray],
    update_progress: Callable[[int], Any],
) -> dict[int, ScenarioResult]:
    scenario_results: dict[int, ScenarioResult] = {}

    for n in ns:
        key_prefix = f"{sc_name}_n{n}"
        tx = jnp.array(dataset_npz[f"{key_prefix}_tx"][:batch], dtype=jnp.float32)
        rx = jnp.array(dataset_npz[f"{key_prefix}_rx"][:batch], dtype=jnp.float32)
        origins = jnp.array(dataset_npz[f"{key_prefix}_origins"][:batch], dtype=jnp.float32)
        vectors = jnp.array(dataset_npz[f"{key_prefix}_vectors"][:batch], dtype=jnp.float32)
        itypes = jnp.array(dataset_npz[f"{key_prefix}_itypes"][:batch], dtype=jnp.int32)
        expected = jnp.array(dataset_npz[f"{key_prefix}_expected"][:batch], dtype=jnp.float32)

        if sc_name == "C":  # Reflection-only
            ref_label = "image-ref"
            t_ref_ms = _bench_with_timeit(
                lambda: _run_trace_rays(
                    tx,
                    rx,
                    origins,
                    vectors,
                    itypes,
                    use_image_method=True,
                    num_iters=0,
                    num_iters_linesearch=1,
                ),
                repeats=repeats,
            )
            update_progress(1)
        else:
            ref_label = "ecos-ref"
            t_ref_ms = None

        per_solver_xs: dict[str, list[float]] = {}
        per_solver_ys: dict[str, list[float]] = {}

        for cfg in solver_cfgs:
            xs_ms: list[float] = []
            ys_err: list[float] = []

            for iters in iters_grid:
                got = _run_trace_rays(
                    tx,
                    rx,
                    origins,
                    vectors,
                    itypes,
                    use_image_method=cfg.use_image_method,
                    num_iters=iters,
                    num_iters_linesearch=cfg.num_iters_linesearch,
                )
                t_ms = _bench_with_timeit(
                    lambda cfg=cfg, iters=iters: _run_trace_rays(
                        tx,
                        rx,
                        origins,
                        vectors,
                        itypes,
                        use_image_method=cfg.use_image_method,
                        num_iters=iters,
                        num_iters_linesearch=cfg.num_iters_linesearch,
                    ),
                    repeats=repeats,
                )
                err = _mean_error(got, expected)
                xs_ms.append(t_ms)
                ys_err.append(err)
                update_progress(1)

            per_solver_xs[cfg.name] = xs_ms
            per_solver_ys[cfg.name] = ys_err

        scenario_results[n] = {
            "ref_time_ms": t_ref_ms,
            "ref_label": ref_label,
            "xs": per_solver_xs,
            "ys": per_solver_ys,
        }

    return scenario_results


def main():
    args = _parse_args()
    dataset_path = args.dataset
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset {dataset_path} not found. Please run scripts/generate_benchmark_dataset.py first.")

    dataset_npz = np.load(dataset_path)

    batch = args.batch
    ns = args.ns
    iters_grid = tuple(args.iters)
    repeats = args.repeats
    linesearch_iters = args.linesearch_iters

    solver_cfgs = [
        SolverConfig("image-method", True, linesearch_iters, "-", "o", "royalblue"),
        SolverConfig("bfgs", False, linesearch_iters, "-", "d", "dimgray"),
    ]
    all_scenarios = {
        "A": ("A", 0.0, 1),  # diffraction-only, 1D
        "B": ("B", 0.0, 2),  # diffraction-only, 2D
        "C": ("C", 1.0, 2),  # reflection-only, 2D
        "D": ("D", 0.5, 2),  # mixed, 2D
    }
    scenarios = [all_scenarios[k] for k in args.scenarios]

    total_solver_tasks = len(scenarios) * len(ns) * len(solver_cfgs) * len(iters_grid)
    total_ref_tasks = sum(len(ns) for sc_name, _, _ in scenarios if sc_name == "C")
    total_tasks = total_solver_tasks + total_ref_tasks

    fig, axes = plt.subplots(
        len(scenarios), len(ns), figsize=(16, 3 * len(scenarios)), sharey="row"
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]], dtype=object)
    elif axes.ndim == 1:
        if len(scenarios) == 1:
            axes = axes[np.newaxis, :]
        else:
            axes = axes[:, np.newaxis]

    platform = jax.default_backend().upper()
    pbar = tqdm(total=total_tasks, desc=f"[{platform}] Benchmark progress", unit="task")

    for row, (sc_name, p_reflect, num_dims) in enumerate(scenarios):
        scenario_results = run_benchmark(
            sc_name=sc_name,
            p_reflect=p_reflect,
            num_dims=num_dims,
            ns=ns,
            iters_grid=iters_grid,
            batch=batch,
            repeats=repeats,
            solver_cfgs=solver_cfgs,
            dataset_npz=dataset_npz,
            update_progress=pbar.update,
        )

        for col, n in enumerate(ns):
            ax = axes[row, col]
            result = scenario_results[n]
            xs = result["xs"]
            ys = result["ys"]

            for cfg in solver_cfgs:
                if sc_name == "C" and cfg.use_image_method:
                    continue
                xs_clean = [x for x, y in zip(xs[cfg.name], ys[cfg.name]) if np.isfinite(x) and np.isfinite(y) and y > 0 and x > 0]
                ys_clean = [y for x, y in zip(xs[cfg.name], ys[cfg.name]) if np.isfinite(x) and np.isfinite(y) and y > 0 and x > 0]
                if xs_clean and ys_clean:
                    ax.plot(
                        xs_clean,
                        ys_clean,
                        cfg.line_style,
                        marker=cfg.marker,
                        color=cfg.color,
                        alpha=0.75,
                        label=cfg.name if (row == 0 and col == 0) else None,
                    )

            if result["ref_time_ms"] is not None and np.isfinite(result["ref_time_ms"]):
                ax.axvline(
                    result["ref_time_ms"],
                    color="black",
                    linestyle=(0, (2, 2)),
                    linewidth=2.5,
                    label=result["ref_label"] if (row == 2 and col == 0) else None,
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(alpha=0.75, which="both")
            ax.set_title(f"p={p_reflect}, dims={num_dims}, n={n}")
            if row == len(scenarios) - 1:
                ax.set_xlabel("Execution time (ms)")
            if col == 0:
                ax.set_ylabel("Average error on X*")

    handles_0, labels_0 = axes[0, 0].get_legend_handles_labels()
    if len(scenarios) > 2:
        handles_ref, labels_ref = axes[2, 0].get_legend_handles_labels()
        handles = handles_0 + handles_ref
        labels = labels_0 + labels_ref
    else:
        handles, labels = handles_0, labels_0

    precision = "f64" if jnp.ones(()).dtype == jnp.float64 else "f32"
    fig.suptitle(f"Trace Rays Benchmark ({platform}, {precision}) vs ECOS Ground Truth")
    fig.legend(handles, labels, loc="upper right", ncol=4, frameon=True)
    fig.tight_layout()
    pbar.close()

    out_path = Path("benchmarks") / f"trace_rays_benchmark_{platform.lower()}_{precision}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200)
    print(f"Benchmark plot saved to {out_path}")


if __name__ == "__main__":
    main()
