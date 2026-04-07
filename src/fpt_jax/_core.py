from functools import partial

import jax
import jax.numpy as jnp


@partial(
    jax.jit,
    static_argnames=(
        "num_iters",
        "unroll",
        "num_iters_linesearch",
        "unroll_linesearch",
        "implicit_diff",
        "solver",
    ),
)
def trace_rays(
    tx: jax.Array,
    rx: jax.Array,
    object_origins: jax.Array,
    object_vectors: jax.Array,
    *,
    solver: str = "bfgs",
    num_iters: int,
    unroll: int | bool = 1,
    num_iters_linesearch: int = 1,
    unroll_linesearch: int | bool = 1,
    implicit_diff: bool = True,
) -> jax.Array:
    """
    Compute the points of interaction of rays with objects using Fermat's principle.

    Each ray is obtained by minimizing the total travel distance from transmitter to receiver
    using one of the supported solvers. At each iteration, a line search is performed
    to find the optimal step size along the descent direction.

    This function accepts batched inputs, where the leading dimensions must be broadcast-compatible.

    Args:
        tx: Transmitter positions of shape `(..., 3)`.
        rx: Receiver positions of shape `(..., 3)`.
        object_origins: Origins of the objects of shape `(..., num_interactions, 3)`.
        object_vectors: Vectors defining the objects of shape `(..., num_interactions, num_dims, 3)`.
        num_iters: Number of iterations for the optimization algorithm.
        unroll: If an integer, the number of optimization iterations to unroll in the JAX `scan`.
            If `True`, unroll all iterations. If `False`, do not unroll.
        num_iters_linesearch: Number of iterations for the line search fixed-point iteration.
        unroll_linesearch: If an integer, the number of fixed-point iterations to unroll in the JAX `scan`.
            If `True`, unroll all iterations. If `False`, do not unroll.
        implicit_diff: Whether to use implicit differentiation for computing the gradient.
            If `True`, assumes that the solution has converged and applies the implicit function theorem
            to differentiate the optimization problem with respect to the input parameters:
            `tx`, `rx`, `object_origins`, and `object_vectors`.
            If `False`, the gradient is computed by backpropagating through all iterations of the optimization algorithm.

            Using implicit differentiation is more memory- and computationally efficient, as it does not require storing
            intermediate values from all iterations, but it may be less accurate if the optimization has not fully
            converged. Moreover, implicit differentiation is not compatible with forward-mode autodiff in JAX.
        solver: The optimization algorithm to use. Supported values are "bfgs" and "ecos".

    Returns:
        The points of interaction of shape `(..., num_interactions, 3)`.
        To include the transmitter and receiver positions, concatenate `tx` and `rx` to the result.
    """
    if solver == "image-method":
        from ._image_method import trace_ray as trace_ray_fn
    elif solver == "bfgs":
        from ._bfgs import trace_ray as trace_ray_fn
    elif solver == "optimistix-bfgs":
        from ._optimistix import trace_ray_bfgs as trace_ray_fn
    elif solver == "ecos":
        from ._ecos import trace_ray as trace_ray_fn
    elif solver == "cvxpy":
        from ._cvxpy import trace_ray as trace_ray_fn
    else:
        raise ValueError(f"Unsupported solver: {solver!r}")

    return jnp.vectorize(
        partial(
            trace_ray_fn,
            num_iters=num_iters,
            unroll=unroll,
            num_iters_linesearch=num_iters_linesearch,
            unroll_linesearch=unroll_linesearch,
            implicit_diff=implicit_diff,
        ),
        signature="(3),(3),(n,3),(n,d,3)->(n,3)",
    )(tx, rx, object_origins, object_vectors)
