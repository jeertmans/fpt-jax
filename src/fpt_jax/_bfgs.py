from typing import NamedTuple

import jax
import jax.numpy as jnp

from ._common import t_to_xyz


def _fixed_point_fn(
    alpha: jax.Array,
    p: jax.Array,
    t: jax.Array,
    tx: jax.Array,
    rx: jax.Array,
    object_origins: jax.Array,
    object_vectors: jax.Array,
) -> jax.Array:
    # We compute the derivative of length(t + alpha * p) w.r.t. alpha and set it to zero
    # This gives us a fixed point equation for alpha:
    # alpha = - sum(dAp^T @ dAt / den) / sum(dAp^T @ dAp / den)
    num_interactions, num_dims, _ = object_vectors.shape
    t = t.reshape(num_interactions, num_dims)
    p = p.reshape(num_interactions, num_dims)
    pad_width = ((1, 1), (0, 0))
    # A @ t
    At = jnp.einsum(
        "ndk,nd->nk", object_vectors, t, precision=jax.lax.Precision.HIGHEST
    )
    At = jnp.pad(At, pad_width, mode="constant", constant_values=0.0)
    # A @ p
    Ap = jnp.einsum(
        "ndk,nd->nk", object_vectors, p, precision=jax.lax.Precision.HIGHEST
    )
    Ap = jnp.pad(Ap, pad_width, mode="constant", constant_values=0.0)
    # b
    b = jnp.concat((tx[None, :], object_origins, rx[None, :]), axis=-2)
    # Deltas
    dAt = jnp.diff(At, axis=-2)
    dAp = jnp.diff(Ap, axis=-2)
    db = jnp.diff(b, axis=-2)
    dX = dAt + db
    # dAp^T @ dX
    num_1 = jnp.sum(dAp * dX, axis=-1)
    # dAp^T @ dAp
    num_2 = jnp.sum(dAp * dAp, axis=-1)
    den = jnp.linalg.norm(dX + alpha * dAp, axis=-1)
    # When the segment length is zero,
    # we ignore the contribution of that segment to the derivative.
    zero_den = den == 0.0

    num_1: jax.Array = jnp.where(zero_den, 0.0, num_1)
    num_2: jax.Array = jnp.where(zero_den, 0.0, num_2)
    den: jax.Array = jnp.where(zero_den, 1.0, den)

    left = jnp.sum(num_1 / den, axis=-1)
    right = jnp.sum(num_2 / den, axis=-1)
    zero_right = right == 0.0
    right = jnp.where(zero_right, 1.0, right)

    return jnp.where(zero_right, alpha, -left / right)


def _grad_t(
    t: jax.Array,
    tx: jax.Array,
    rx: jax.Array,
    object_origins: jax.Array,
    object_vectors: jax.Array,
) -> jax.Array:
    # We compute the gradient of length(t) w.r.t. t
    num_interactions, num_dims, _ = object_vectors.shape
    t = t.reshape(num_interactions, num_dims)
    pad_width = ((1, 1), (0, 0))
    # A @ t
    At = jnp.einsum(
        "ndk,nd->nk", object_vectors, t, precision=jax.lax.Precision.HIGHEST
    )
    At = jnp.pad(At, pad_width, mode="constant", constant_values=0.0)
    # b
    b = jnp.concat((tx[None, :], object_origins, rx[None, :]), axis=-2)
    # Deltas
    dAt = jnp.diff(At, axis=-2)
    db = jnp.diff(b, axis=-2)
    dX = dAt + db

    den = jnp.linalg.norm(dX, axis=-1, keepdims=True)
    # When the segment length is zero,
    # we ignore the contribution of that segment to the derivative
    zero_den = den == 0.0
    den: jax.Array = jnp.where(zero_den, 1.0, den)
    num = dX  # dX is zero where den is zero

    num_den = num / den

    left = jnp.einsum(
        "ndk,nk->nd",
        object_vectors,
        num_den[:-1, :],
        precision=jax.lax.Precision.HIGHEST,
    )
    right = jnp.einsum(
        "ndk,nk->nd",
        object_vectors,
        num_den[+1:, :],
        precision=jax.lax.Precision.HIGHEST,
    )
    return (left - right).ravel()


class _State(NamedTuple):
    t: jax.Array
    params: tuple[jax.Array, ...]
    grad: jax.Array
    H: jax.Array


def _linesearch(
    p: jax.Array,
    t: jax.Array,
    args: tuple[jax.Array, ...],
    *,
    num_iters: int,
    unroll: int | bool,
) -> jax.Array:
    init_alpha = jnp.array(1.0)
    return jax.lax.scan(
        lambda alpha, _: (_fixed_point_fn(alpha, p, t, *args), None),
        init_alpha,
        xs=None,
        length=num_iters,
        unroll=unroll,
    )[0]


def _step_fn(
    state: _State, num_iters_linesearch: int, unroll_linesearch: int | bool
) -> _State:
    p = -state.H @ state.grad
    alpha = _linesearch(
        p,
        state.t,
        state.params,
        num_iters=num_iters_linesearch,
        unroll=unroll_linesearch,
    )  # Linesearch
    s = alpha * p
    t = state.t + s

    grad = _grad_t(t, *state.params)
    y = grad - state.grad
    Hy = jnp.matmul(state.H, y, precision=jax.lax.Precision.HIGHEST)
    yTHy = jnp.dot(y, Hy, precision=jax.lax.Precision.HIGHEST)
    sTy = jnp.dot(s, y, precision=jax.lax.Precision.HIGHEST)
    ssT = jnp.tensordot(s, s, axes=0, precision=jax.lax.Precision.HIGHEST)
    HysT = jnp.tensordot(Hy, s, axes=0, precision=jax.lax.Precision.HIGHEST)
    syTH = jnp.tensordot(s, Hy, axes=0, precision=jax.lax.Precision.HIGHEST)

    # Skip update if sTy is too small (occurs at first step or when converged)
    skip_update = sTy < jnp.finfo(sTy.dtype).eps
    sTy = jnp.where(skip_update, 1.0, sTy)
    H = jnp.where(
        skip_update,
        state.H,
        state.H + ((sTy + yTHy) * ssT) / (sTy**2) - (HysT + syTH) / sTy,
    )
    return _State(t, state.params, grad, H)


def _solve_t(
    tx: jax.Array,
    rx: jax.Array,
    object_origins: jax.Array,
    object_vectors: jax.Array,
    num_iters: int,
    unroll: int | bool,
    num_iters_linesearch: int,
    unroll_linesearch: int | bool,
) -> jax.Array:
    num_interactions, num_dims, _ = object_vectors.shape
    n = num_interactions * num_dims
    dtype = jnp.result_type(tx, rx, object_origins, object_vectors)
    t = jnp.zeros(n, dtype=dtype)
    params = (tx, rx, object_origins, object_vectors)
    grad = jnp.zeros(n, dtype=dtype)
    H = jnp.identity(n, dtype=dtype)
    initial_state = _State(t, params, grad, H)
    final_state = jax.lax.scan(
        lambda state, _: (
            _step_fn(state, num_iters_linesearch, unroll_linesearch),
            None,
        ),
        initial_state,
        xs=None,
        length=num_iters,
        unroll=unroll,
    )[0]
    # Return optimal t, not xyz, as we need t for the backward pass.
    return final_state.t


def _solve_t_fwd(
    tx: jax.Array,
    rx: jax.Array,
    object_origins: jax.Array,
    object_vectors: jax.Array,
    num_iters: int,
    unroll: int | bool,
    num_iters_linesearch: int,
    unroll_linesearch: int | bool,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
    t = _solve_t(
        tx,
        rx,
        object_origins,
        object_vectors,
        num_iters=num_iters,
        unroll=unroll,
        num_iters_linesearch=num_iters_linesearch,
        unroll_linesearch=unroll_linesearch,
    )
    residuals = (
        t,
        tx,
        rx,
        object_origins,
        object_vectors,
    )  # We need to save the inputs for the backward pass.
    return t, residuals


def _solve_t_bwd(
    num_iters: int,
    unroll: int | bool,
    num_iters_linesearch: int,
    unroll_linesearch: int | bool,
    residuals: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    cotangent: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # JAX automatically provides nondiff_args as the first arguments.
    # However, none of them should (in theory) impact the backward pass,
    # as we assume that the optimization procedure has converged.
    del num_iters, unroll, num_iters_linesearch, unroll_linesearch
    t, tx, rx, object_origins, object_vectors = residuals

    def fn_t(t: jax.Array) -> jax.Array:
        return _grad_t(t, tx, rx, object_origins, object_vectors)

    _, vjp_fun_t = jax.vjp(fn_t, t)

    def matvec(u: jax.Array) -> jax.Array:
        return vjp_fun_t(u)[0]

    v = -cotangent
    A = jax.jacfwd(matvec)(jnp.zeros_like(v))
    u = jax.scipy.linalg.solve(A, v, assume_a="pos")

    def fun_args(*args: jax.Array) -> jax.Array:
        return _grad_t(t, *args)

    _, vjp_fun_args = jax.vjp(fun_args, tx, rx, object_origins, object_vectors)

    return vjp_fun_args(u)


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
    solver_fn = _solve_t
    if implicit_diff:
        solver_fn = jax.custom_vjp(
            solver_fn,
            nondiff_argnames=(
                "num_iters",
                "unroll",
                "num_iters_linesearch",
                "unroll_linesearch",
            ),
        )
        solver_fn.defvjp(_solve_t_fwd, _solve_t_bwd)

    t = _solve_t(
        tx,
        rx,
        object_origins,
        object_vectors,
        num_iters=num_iters,
        unroll=unroll,
        num_iters_linesearch=num_iters_linesearch,
        unroll_linesearch=unroll_linesearch,
    )
    return t_to_xyz(t, object_origins, object_vectors)
