import jax
import optimistix as optx
import jax.numpy as jnp
import lineax as lx

from ._common import objective, t_to_xyz


def _solve_t(
    tx: jax.Array,
    rx: jax.Array,
    object_origins: jax.Array,
    object_vectors: jax.Array,
    solver: optx.AbstractMinimiser,
    num_iters: int,
    unroll: int | bool,
    num_iters_linesearch: int,
    unroll_linesearch: int | bool,
    implicit_diff: bool,
) -> jax.Array:
    del unroll, num_iters_linesearch, unroll_linesearch
    num_interactions, num_dims, _ = object_vectors.shape
    n = num_interactions * num_dims
    dtype = jnp.result_type(tx, rx, object_origins, object_vectors)
    t = jnp.zeros(n, dtype=dtype)
    if n == 0:
        return t

    solution = optx.minimise(
        lambda t, args: objective(t, *args),
        solver,
        y0=t,
        args=(tx, rx, object_origins, object_vectors),
        max_steps=num_iters,
        adjoint=optx.ImplicitAdjoint()
        if implicit_diff
        else optx.RecursiveCheckpointAdjoint(),
        tags=frozenset({lx.positive_semidefinite_tag}),
        throw=False,
    )
    return solution.value


def _trace_ray(
    tx: jax.Array,
    rx: jax.Array,
    object_origins: jax.Array,
    object_vectors: jax.Array,
    *,
    solver: optx.AbstractMinimiser,
    num_iters: int,
    unroll: int | bool = 1,
    num_iters_linesearch: int = 1,
    unroll_linesearch: int | bool = 1,
    implicit_diff: bool = True,
) -> jax.Array:
    t = _solve_t(
        tx,
        rx,
        object_origins,
        object_vectors,
        num_iters=num_iters,
        unroll=unroll,
        num_iters_linesearch=num_iters_linesearch,
        unroll_linesearch=unroll_linesearch,
        solver=solver,
        implicit_diff=implicit_diff,
    )
    return t_to_xyz(t, object_origins, object_vectors)


def trace_ray_bfgs(
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
    return _trace_ray(
        tx,
        rx,
        object_origins,
        object_vectors,
        solver=optx.BFGS(atol=1e-16, rtol=1e-16),  # type: ignore[ty:invalid-argument-type]
        num_iters=num_iters,
        unroll=unroll,
        num_iters_linesearch=num_iters_linesearch,
        unroll_linesearch=unroll_linesearch,
        implicit_diff=implicit_diff,
    )
