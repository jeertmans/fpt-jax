import cvxpy as cp
import jax
import jax.numpy as jnp

from cvxpylayers.jax import CvxpyLayer

from ._common import t_to_xyz


def trace_ray(
    tx: jax.Array,
    rx: jax.Array,
    object_origins: jax.Array,
    object_vectors: jax.Array,
    *,
    num_iters: int,
    unroll: int | bool = 1,
    num_iters_linesearch: int = 1,
    unroll_linesearch: int | bool = 1,
    implicit_diff: bool = True,
) -> jax.Array:
    del unroll, num_iters_linesearch, unroll_linesearch, implicit_diff

    num_interactions, num_dims, _ = object_vectors.shape
    dtype = jnp.result_type(tx, rx, object_origins, object_vectors)

    def solve(A_jax: jax.Array, b_jax: jax.Array) -> jax.Array:
        t = cp.Variable((num_interactions, num_dims))
        s = cp.Variable(num_interactions + 1)
        A = cp.Parameter((num_interactions + 2, num_dims, 3))
        b = cp.Parameter((num_interactions + 2, 3))

        # Helper to calculate the 3D position of an interaction point i
        def get_x(i):
            if i == 0:
                return b[0]  # TX
            elif i == num_interactions + 1:
                return b[num_interactions + 1]  # RX
            else:
                return t[i - 1] @ A[i] + b[i]  # Interaction point

        # 3. Objective and Constraints
        objective = cp.Minimize(cp.sum(s))

        # cp.SOC(s_i, vector) enforces ||vector||_2 <= s_i
        constraints = [
            cp.SOC(s[i], get_x(i + 1) - get_x(i)) for i in range(num_interactions + 1)
        ]
        problem = cp.Problem(
            objective,
            constraints,  # type: ignore[ty:invalid-argument-type]
        )
        assert problem.is_dpp()

        layer = CvxpyLayer(
            problem,
            parameters=[A, b],
            variables=[t, s],
            solver_args={"max_iters": num_iters, "verbose": False},
        )

        t_opt, _ = layer(A_jax, b_jax)
        return t_opt.ravel()

    zeros = jnp.zeros((1, num_dims, 3))
    A = jnp.concatenate([zeros, object_vectors, zeros], axis=0)
    b = jnp.concatenate([tx[None, :], object_origins, rx[None, :]], axis=0)

    t = jax.pure_callback(
        solve,
        jax.ShapeDtypeStruct((num_interactions * num_dims,), dtype),
        A,
        b,
        vmap_method="sequential",
    )

    return t_to_xyz(t, object_origins, object_vectors)
