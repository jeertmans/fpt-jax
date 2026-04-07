"""Benchmark script for `trace_rays` solvers.

Usage:
    uv run --group benchmarks benchmarks/trace_rays.py
"""

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from fpt_jax import trace_rays


@dataclass(frozen=True)
class SolverConfig:
    name: str
    solver: str
    num_iters_linesearch: int
    line_style: str
    marker: str
    color: str


class ScenarioResult(TypedDict):
    ref_time_ms: float | None
    ref_label: str
    xs: dict[str, list[float]]
    ys: dict[str, list[float]]


def random_planes(
    key: jax.Array,
    shape: tuple[int, ...],
) -> tuple[jax.Array, jax.Array]:
    o, u, w = jr.normal(key, (3, *shape, 3))
    u /= jnp.linalg.norm(u, axis=-1, keepdims=True)
    w /= jnp.linalg.norm(w, axis=-1, keepdims=True)
    v = jnp.cross(w, u)
    v /= jnp.linalg.norm(v, axis=-1, keepdims=True)
    return o, jnp.stack((u, v), axis=-2)


def random_edges(
    key: jax.Array,
    shape: tuple[int, ...],
) -> tuple[jax.Array, jax.Array]:
    o, d = jr.normal(key, (2, *shape, 3))
    d /= jnp.linalg.norm(d, axis=-1, keepdims=True)
    z = jnp.zeros_like(d)
    return o, jnp.stack((d, z), axis=-2)


def _sample_case(
    *,
    batch: int,
    n: int,
    reflection_over_diffraction_p: float,
    num_dims: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    key_interactions, key_planes, key_edges, key_txs, key_rxs = jr.split(key, 5)
    is_reflection = jr.bernoulli(
        key_interactions,
        p=reflection_over_diffraction_p,
        shape=(batch, n),
    )

    plane_origins, plane_vectors = random_planes(key_planes, (batch, n))
    edge_origins, edge_vectors = random_edges(key_edges, (batch, n))

    object_origins = jnp.where(is_reflection[..., None], plane_origins, edge_origins)
    object_vectors = jnp.where(
        is_reflection[..., None, None],
        plane_vectors,
        edge_vectors,
    )
    if num_dims == 1:
        object_vectors = object_vectors[..., :1, :]

    tx = jr.normal(key_txs, (batch, 3))
    rx = jr.normal(key_rxs, (batch, 3))
    return tx, rx, object_origins, object_vectors


def _mean_error(got: jax.Array, expected: jax.Array) -> float:
    # Average point-to-point Euclidean error on interaction points X*.
    err = jnp.linalg.norm(got - expected, axis=-1).mean(axis=-1)
    return float(jnp.mean(err))


def _masked_mean_error(
    got: jax.Array,
    expected: jax.Array,
    valid: jax.Array,
) -> float:
    # Average error only over valid image-method reflection paths.
    per_case = jnp.linalg.norm(got - expected, axis=-1).mean(axis=-1)
    valid = valid.astype(per_case.dtype)
    total = jnp.sum(per_case * valid)
    count = jnp.sum(valid)
    return float(jnp.where(count > 0, total / count, jnp.nan))


def valid_reflection_paths(
    plane_origins: jax.Array,
    plane_vectors: jax.Array,
    ray_paths: jax.Array,
) -> jax.Array:
    all_finite = jnp.isfinite(ray_paths).all(axis=(-1, -2))

    d_prev = ray_paths[..., :-2, :] - plane_origins
    d_next = ray_paths[..., +2:, :] - plane_origins

    plane_normals = jnp.cross(plane_vectors[..., 0, :], plane_vectors[..., 1, :])

    dot_prev = jnp.sum(d_prev * plane_normals, axis=-1)
    dot_next = jnp.sum(d_next * plane_normals, axis=-1)
    same_sign = (jnp.sign(dot_prev) == jnp.sign(dot_next)).all(axis=-1)
    return all_finite & same_sign


def _bench_with_timeit(fn: Callable[[], jax.Array], *, repeats: int) -> float:
    times_ms = []
    _ = fn().block_until_ready()  # Warm-up to trigger JIT compilation before timing.
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
    *,
    solver: str,
    num_iters: int,
    num_iters_linesearch: int,
) -> jax.Array:
    y = trace_rays(
        tx,
        rx,
        object_origins,
        object_vectors,
        num_iters=num_iters,
        num_iters_linesearch=num_iters_linesearch,
        solver=solver,
    )
    return y.block_until_ready()


def _parse_args() -> argparse.Namespace:
    default_batch = 128 if jax.default_backend() == "gpu" else 64
    parser = argparse.ArgumentParser(description="Benchmark `trace_rays` solvers")
    parser.add_argument(
        "--batch",
        type=int,
        default=default_batch,
        help="Batch size per benchmark case.",
    )
    parser.add_argument(
        "--ns",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Values of n (number of interactions) to benchmark.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32],
        help="Iteration counts used for all benchmarked solvers.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of timing repeats (minimum runtime retained).",
    )
    parser.add_argument(
        "--ref-iters",
        type=int,
        default=50,
        help="Number of iterations used for the non-image reference solution.",
    )
    parser.add_argument(
        "--linesearch-iters",
        type=int,
        default=16,
        help="Line-search iterations for non-reference solvers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="PRNG seed used for data generation.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["A", "B", "C", "D"],
        default=["A", "B", "C", "D"],
        help=(
            "Scenarios to run (default: all). "
            "A=diffraction-only(1D), B=diffraction-only(2D), "
            "C=reflection-only(2D), D=mixed(2D)"
        ),
    )
    return parser.parse_args()


def run_benchmark(
    *,
    reflection_over_diffraction_p: float,
    num_dims: int,
    ns: list[int],
    iters_grid: tuple[int, ...],
    batch: int,
    ref_iters: int,
    repeats: int,
    seed: int,
    solver_cfgs: list[SolverConfig],
    update_progress: Callable[[int], Any],
) -> dict[int, ScenarioResult]:
    scenario_results: dict[int, ScenarioResult] = {}

    for n in ns:
        tx, rx, object_origins, object_vectors = _sample_case(
            batch=batch,
            n=n,
            reflection_over_diffraction_p=reflection_over_diffraction_p,
            num_dims=num_dims,
            key=jr.PRNGKey(seed + 100 * num_dims + n),
        )

        if num_dims == 2 and reflection_over_diffraction_p == 1.0:
            reference = _run_trace_rays(
                tx,
                rx,
                object_origins,
                object_vectors,
                solver="image-method",
                num_iters=ref_iters,
                num_iters_linesearch=1,
            )
            image_paths = jnp.concatenate(
                [tx[:, None, :], reference, rx[:, None, :]],
                axis=1,
            )
            valid = valid_reflection_paths(object_origins, object_vectors, image_paths)
            ref_solver = "image-method"
            ref_label = "image-ref"
            t_ref_ms = _bench_with_timeit(
                lambda ref_solver=ref_solver: _run_trace_rays(
                    tx,
                    rx,
                    object_origins,
                    object_vectors,
                    solver=ref_solver,
                    num_iters=ref_iters,
                    num_iters_linesearch=1,
                ),
                repeats=repeats,
            )
            update_progress(1)
        else:
            reference = _run_trace_rays(
                tx,
                rx,
                object_origins,
                object_vectors,
                solver="ecos",
                num_iters=ref_iters,
                num_iters_linesearch=1,
            )
            valid = None
            ref_solver = "ecos"
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
                    object_origins,
                    object_vectors,
                    solver=cfg.solver,
                    num_iters=iters,
                    num_iters_linesearch=cfg.num_iters_linesearch,
                )
                t_ms = _bench_with_timeit(
                    lambda cfg=cfg, iters=iters: _run_trace_rays(
                        tx,
                        rx,
                        object_origins,
                        object_vectors,
                        solver=cfg.solver,
                        num_iters=iters,
                        num_iters_linesearch=cfg.num_iters_linesearch,
                    ),
                    repeats=repeats,
                )
                err = (
                    _masked_mean_error(got, reference, valid)
                    if valid is not None
                    else _mean_error(got, reference)
                )
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
    batch = args.batch
    ns = args.ns
    iters_grid = tuple(args.iters)
    repeats = args.repeats
    ref_iters = args.ref_iters
    linesearch_iters = args.linesearch_iters
    seed = args.seed

    solver_cfgs = [
        SolverConfig("ecos", "ecos", linesearch_iters, "-", "o", "royalblue"),
        SolverConfig("bfgs", "bfgs", linesearch_iters, "-", "d", "dimgray"),
        SolverConfig(
            "optimistix-bfgs",
            "optimistix-bfgs",
            linesearch_iters,
            "--",
            "s",
            "peru",
        ),
    ]
    all_scenarios = {
        "A": (0.0, 1),  # diffraction-only, 1D
        "B": (0.0, 2),  # diffraction-only, 2D
        "C": (1.0, 2),  # reflection-only, 2D
        "D": (0.5, 2),  # mixed, 2D
    }
    scenarios = [all_scenarios[k] for k in args.scenarios]

    total_solver_tasks = len(scenarios) * len(ns) * len(solver_cfgs) * len(iters_grid)
    total_ref_tasks = sum(
        len(ns)
        for p_reflect, num_dims in scenarios
        if num_dims == 2 and p_reflect == 1.0
    )
    total_tasks = total_solver_tasks + total_ref_tasks

    fig, axes = plt.subplots(len(scenarios), len(ns), figsize=(16, 12), sharey="row")
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]], dtype=object)
    elif axes.ndim == 1:
        if len(scenarios) == 1:
            axes = axes[np.newaxis, :]
        else:
            axes = axes[:, np.newaxis]

    pbar = tqdm(total=total_tasks, desc="Benchmark progress", unit="task")

    for row, (p_reflect, num_dims) in enumerate(scenarios):
        scenario_results = run_benchmark(
            reflection_over_diffraction_p=p_reflect,
            num_dims=num_dims,
            ns=ns,
            iters_grid=iters_grid,
            batch=batch,
            ref_iters=ref_iters,
            repeats=repeats,
            seed=seed,
            solver_cfgs=solver_cfgs,
            update_progress=pbar.update,
        )

        for col, n in enumerate(ns):
            ax = axes[row, col]
            result = scenario_results[n]
            xs = result["xs"]
            ys = result["ys"]

            for cfg in solver_cfgs:
                ax.plot(
                    xs[cfg.name],
                    ys[cfg.name],
                    cfg.line_style,
                    marker=cfg.marker,
                    color=cfg.color,
                    alpha=0.75,
                    label=cfg.name if (row == 0 and col == 0) else None,
                )

            if result["ref_time_ms"] is not None:
                ax.axvline(
                    result["ref_time_ms"],
                    color="black",
                    linestyle=(0, (2, 2)),
                    linewidth=2.5,
                    label=result["ref_label"] if (row == 0 and col == 0) else None,
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(alpha=0.75, which="both")
            ax.set_title(f"p={p_reflect}, dims={num_dims}, n={n}")
            if row == len(scenarios) - 1:
                ax.set_xlabel("Execution time (ms)")
            if col == 0:
                ax.set_ylabel("Average error on X*")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    pbar.close()

    out_path = Path("benchmarks") / "trace_rays_benchmark.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200)


if __name__ == "__main__":
    main()
