"""Generate certified ground truth benchmark dataset using ultra-low tolerance ECOS/Clarabel (via CVXPY).

Usage:
    uv run --group benchmarks python scripts/generate_benchmark_dataset.py
"""

import argparse
from pathlib import Path
import cvxpy as cp
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from tqdm.auto import tqdm


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


def solve_ground_truth_ecos(
    tx: np.ndarray,
    rx: np.ndarray,
    origins: np.ndarray,
    vectors: np.ndarray,
    itypes: np.ndarray,
) -> np.ndarray:
    N = len(itypes)
    diffraction_indices = [i for i in range(N) if itypes[i] == 1]
    t_vars = {idx: cp.Variable(1) for idx in diffraction_indices}

    def get_anchor_expr(idx: int):
        if idx == -1:
            return tx
        elif idx == N:
            return rx
        else:
            return origins[idx] + t_vars[idx] * vectors[idx, 0]

    anchor_indices = [-1] + diffraction_indices + [N]

    if len(diffraction_indices) > 0:
        constraints = []
        slack_vars = []
        for seg_idx in range(len(anchor_indices) - 1):
            start_idx = anchor_indices[seg_idx]
            end_idx = anchor_indices[seg_idx + 1]
            intermediate_indices = list(range(start_idx + 1, end_idx))

            start_expr = get_anchor_expr(start_idx)
            end_expr = get_anchor_expr(end_idx)

            curr_mirrored = start_expr
            for pl_idx in intermediate_indices:
                orig_i = origins[pl_idx]
                v0 = vectors[pl_idx, 0]
                v1 = vectors[pl_idx, 1] if vectors.shape[1] > 1 else np.zeros(3)
                n_i = np.cross(v0, v1)
                norm_n = np.linalg.norm(n_i)
                if norm_n < 1e-7:
                    raise ValueError("Degenerate plane")
                n_hat = n_i / norm_n
                if itypes[pl_idx] == 0:  # Reflection
                    d = (curr_mirrored - orig_i) @ n_hat
                    curr_mirrored = curr_mirrored - 2.0 * d * n_hat

            s_k = cp.Variable()
            slack_vars.append(s_k)
            diff = end_expr - curr_mirrored
            constraints.append(cp.SOC(s_k, diff))

        prob = cp.Problem(cp.Minimize(cp.sum(slack_vars)), constraints)
        
        # Try Clarabel at 1e-14
        try:
            prob.solve(
                solver=cp.CLARABEL,
                tol_gap_abs=1e-14,
                tol_gap_rel=1e-14,
                tol_feas=1e-14,
                verbose=False,
            )
        except Exception:
            pass
            
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            prob.solve(
                solver=cp.ECOS,
                abstol=1e-12,
                reltol=1e-12,
                feastol=1e-12,
                verbose=False,
            )
            
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise ValueError(f"Solver status: {prob.status}")

    edge_points = np.zeros((N, 3))
    for idx in range(N):
        if itypes[idx] == 1:
            edge_points[idx] = origins[idx] + t_vars[idx].value * vectors[idx, 0]
        else:
            edge_points[idx] = origins[idx]

    images = []
    curr_src = tx.copy()
    for i in range(N):
        orig_i = origins[i]
        v0 = vectors[i, 0]
        v1 = vectors[i, 1] if vectors.shape[1] > 1 else np.zeros(3)
        n_i = np.cross(v0, v1)
        norm_n = np.linalg.norm(n_i)
        n_hat = n_i / (norm_n if norm_n > 1e-7 else 1.0)

        if itypes[i] == 0:
            d = np.dot(curr_src - orig_i, n_hat)
            curr_src = curr_src - 2.0 * d * n_hat
            images.append(curr_src.copy())
        elif itypes[i] == 2:
            images.append(curr_src.copy())
        else:
            curr_src = edge_points[i].copy()
            images.append(curr_src.copy())

    pts = np.zeros((N, 3))
    curr_dest = rx.copy()
    for i in reversed(range(N)):
        orig_i = origins[i]
        v0 = vectors[i, 0]
        v1 = vectors[i, 1] if vectors.shape[1] > 1 else np.zeros(3)
        n_i = np.cross(v0, v1)
        norm_n = np.linalg.norm(n_i)
        n_hat = n_i / (norm_n if norm_n > 1e-7 else 1.0)
        img_i = images[i]

        if itypes[i] in (0, 2):
            direction = curr_dest - img_i
            denom = np.dot(direction, n_hat)
            if abs(denom) < 0.05:
                raise ValueError("Glancing ray")
            s = np.dot(orig_i - img_i, n_hat) / denom
            pts[i] = img_i + s * direction
            curr_dest = pts[i].copy()
        else:
            pts[i] = edge_points[i].copy()
            curr_dest = pts[i].copy()

    # Physical check
    full_pts = np.concatenate([tx[None, :], pts, rx[None, :]], axis=0)
    for i in range(N):
        if itypes[i] == 0:
            v0 = vectors[i, 0]
            v1 = vectors[i, 1]
            n_hat = np.cross(v0, v1)
            n_hat = n_hat / np.linalg.norm(n_hat)
            d_in = np.dot(full_pts[i] - origins[i], n_hat)
            d_out = np.dot(full_pts[i + 2] - origins[i], n_hat)
            if d_in * d_out < 0:
                raise ValueError("Path crossed plane on reflection")

    if np.max(np.abs(pts)) > 100.0:
        raise ValueError("Point out of bounds")

    return pts


def generate_dataset(
    *,
    batch_size: int = 128,
    ns: list[int] = [1, 2, 3, 4, 5],
    scenarios: dict[str, tuple[float, int]] = {
        "A": (0.0, 1),
        "B": (0.0, 2),
        "C": (1.0, 2),
        "D": (0.5, 2),
    },
    seed: int = 42,
    output_path: Path = Path("benchmarks/data/benchmark_dataset.npz"),
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset: dict[str, np.ndarray] = {}

    total_configs = len(scenarios) * len(ns)
    pbar = tqdm(total=total_configs * batch_size, desc="Generating certified cases (tol=1e-14)")

    key = jr.PRNGKey(seed)

    for sc_name, (p_reflect, num_dims) in scenarios.items():
        for n in ns:
            tx_list = []
            rx_list = []
            orig_list = []
            vec_list = []
            itype_list = []
            pts_list = []

            while len(tx_list) < batch_size:
                key, subkey = jr.split(key)
                tx, rx = jr.normal(subkey, (2, 3))
                tx = np.array(tx)
                rx = np.array(rx)

                key, subkey = jr.split(key)
                is_refl = jr.bernoulli(subkey, p=p_reflect, shape=(n,))
                itypes = np.where(is_refl, 0, 1)

                key, subkey = jr.split(key)
                p_orig, p_vec = random_planes(subkey, (n,))
                key, subkey = jr.split(key)
                e_orig, e_vec = random_edges(subkey, (n,))

                origins = np.where(is_refl[:, None], p_orig, e_orig)
                vectors = np.where(is_refl[:, None, None], p_vec, e_vec)
                if num_dims == 1:
                    vectors = vectors[:, :1, :]

                origins = np.array(origins)
                vectors = np.array(vectors)

                try:
                    pts = solve_ground_truth_ecos(tx, rx, origins, vectors, itypes)
                    tx_list.append(tx)
                    rx_list.append(rx)
                    orig_list.append(origins)
                    vec_list.append(vectors)
                    itype_list.append(itypes)
                    pts_list.append(pts)
                    pbar.update(1)
                except Exception:
                    continue

            key_prefix = f"{sc_name}_n{n}"
            dataset[f"{key_prefix}_tx"] = np.stack(tx_list)
            dataset[f"{key_prefix}_rx"] = np.stack(rx_list)
            dataset[f"{key_prefix}_origins"] = np.stack(orig_list)
            dataset[f"{key_prefix}_vectors"] = np.stack(vec_list)
            dataset[f"{key_prefix}_itypes"] = np.stack(itype_list)
            dataset[f"{key_prefix}_expected"] = np.stack(pts_list)

    pbar.close()
    np.savez_compressed(output_path, **dataset)
    print(f"Saved certified benchmark dataset to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate certified benchmark dataset with ultra-low tolerance")
    parser.add_argument("--batch", type=int, default=128, help="Batch size per (scenario, n)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/data/benchmark_dataset.npz"), help="Output path")
    args = parser.parse_args()

    generate_dataset(batch_size=args.batch, seed=args.seed, output_path=args.output)


if __name__ == "__main__":
    main()
