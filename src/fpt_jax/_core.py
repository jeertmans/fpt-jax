from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

## Private API ##


class _State(NamedTuple):
    t: jax.Array
    params: tuple[jax.Array, ...]
    grad: jax.Array
    H: jax.Array


def _planar_info(object_vectors: jax.Array) -> tuple[jax.Array, jax.Array]:
    num_interactions, num_dims, _ = object_vectors.shape
    if num_dims < 2:
        is_planar = jnp.zeros((num_interactions,), dtype=bool)
        n_hat = jnp.zeros((num_interactions, 3), dtype=object_vectors.dtype)
    else:
        v0 = object_vectors[:, 0, :]
        v1 = object_vectors[:, 1, :]
        normal = jnp.cross(v0, v1)
        norm = jnp.linalg.norm(normal, axis=-1, keepdims=True)
        safe_norm = jnp.where(norm == 0.0, 1.0, norm)
        n_hat = normal / safe_norm
        is_planar = norm[:, 0] > 1e-7
    return is_planar, n_hat


def _t_to_xyz_general(
    t: jax.Array, object_origins: jax.Array, object_vectors: jax.Array
) -> jax.Array:
    num_interactions, num_dims, _ = object_vectors.shape
    if num_dims == 0:
        return object_origins
    t = t.reshape(num_interactions, num_dims)
    return object_origins + jnp.einsum(
        "...nd,...ndk->...nk", t, object_vectors, precision=jax.lax.Precision.HIGHEST
    )


def _t_to_xyz_image_method(
    t: jax.Array,
    tx: jax.Array,
    rx: jax.Array,
    object_origins: jax.Array,
    object_vectors: jax.Array,
    is_planar: jax.Array,
    n_hat: jax.Array,
    interaction_types: jax.Array | None = None,
) -> jax.Array:
    num_interactions, num_dims, _ = object_vectors.shape
    if num_dims > 0:
        edge_points = object_origins + t[:, None] * object_vectors[:, 0, :]
    else:
        edge_points = object_origins

    if interaction_types is not None:
        is_refl = (interaction_types == 0) & is_planar
    else:
        is_refl = is_planar

    def forward_step(
        curr_src: jax.Array, xs: tuple[jax.Array, ...]
    ) -> tuple[jax.Array, jax.Array]:
        orig_i, n_hat_i, is_pl_i, is_rf_i, edge_pt_i = xs
        d_src = jnp.dot(curr_src - orig_i, n_hat_i)
        mirror_shift = jnp.where(is_rf_i, 2.0 * d_src, 0.0)
        mirrored_src = curr_src - mirror_shift * n_hat_i

        next_src = jnp.where(is_pl_i, mirrored_src, edge_pt_i)
        saved_src = jnp.where(is_pl_i, mirrored_src, curr_src)
        return next_src, saved_src

    _, images = jax.lax.scan(
        forward_step,
        tx,
        (object_origins, n_hat, is_planar, is_refl, edge_points),
    )

    def backward_step(
        curr_dest: jax.Array, xs: tuple[jax.Array, ...]
    ) -> tuple[jax.Array, jax.Array]:
        orig_i, n_hat_i, is_pl_i, edge_pt_i, img_i = xs
        direction = curr_dest - img_i
        denom = jnp.dot(direction, n_hat_i)
        sign_denom = jnp.where(denom >= 0.0, 1.0, -1.0)
        safe_denom = jnp.where(jnp.abs(denom) < 1e-4, sign_denom * 1e-4, denom)
        s = jnp.dot(orig_i - img_i, n_hat_i) / safe_denom
        s = jnp.clip(s, -500.0, 500.0)
        intersect_pt = img_i + s * direction

        pt = jnp.where(is_pl_i, intersect_pt, edge_pt_i)
        return pt, pt

    _, pts = jax.lax.scan(
        backward_step,
        rx,
        (object_origins, n_hat, is_planar, edge_points, images),
        reverse=True,
    )
    return pts


def _grad_fn_general(
    t: jax.Array,
    tx: jax.Array,
    rx: jax.Array,
    object_origins: jax.Array,
    object_vectors: jax.Array,
) -> jax.Array:
    num_interactions, num_dims, _ = object_vectors.shape
    if num_dims == 0:
        return jnp.zeros(0, dtype=t.dtype)
    t = t.reshape(num_interactions, num_dims)
    pad_width = ((1, 1), (0, 0))
    At = jnp.einsum(
        "ndk,nd->nk", object_vectors, t, precision=jax.lax.Precision.HIGHEST
    )
    At = jnp.pad(At, pad_width, mode="constant", constant_values=0.0)
    b = jnp.concat((tx[None, :], object_origins, rx[None, :]), axis=-2)
    dAt = jnp.diff(At, axis=-2)
    db = jnp.diff(b, axis=-2)
    dX = dAt + db

    den = jnp.linalg.norm(dX, axis=-1, keepdims=True)
    zero_den = den == 0.0
    den = jnp.where(zero_den, 1.0, den)
    num = dX
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


def _grad_fn_im(
    xyz: jax.Array,
    tx: jax.Array,
    rx: jax.Array,
    object_vectors: jax.Array,
    is_planar: jax.Array,
) -> jax.Array:
    num_interactions, num_dims, _ = object_vectors.shape
    if num_dims == 0:
        return jnp.zeros(num_interactions, dtype=xyz.dtype)
    full = jnp.concat([tx[None, :], xyz, rx[None, :]], axis=0)
    diff = jnp.diff(full, axis=0)
    norms = jnp.linalg.norm(diff, axis=-1, keepdims=True)
    zero_norm = norms == 0.0
    safe_norms = jnp.where(zero_norm, 1.0, norms)
    unit_dirs = diff / safe_norms
    g_xyz = unit_dirs[:-1] - unit_dirs[1:]
    g_t = jnp.sum(g_xyz * object_vectors[:, 0, :], axis=-1)
    return jnp.where(is_planar, 0.0, g_t)


def _fixed_point_fn_general(
    alpha: jax.Array,
    p: jax.Array,
    t: jax.Array,
    tx: jax.Array,
    rx: jax.Array,
    object_origins: jax.Array,
    object_vectors: jax.Array,
) -> jax.Array:
    num_interactions, num_dims, _ = object_vectors.shape
    if num_dims == 0:
        return alpha
    t = t.reshape(num_interactions, num_dims)
    p = p.reshape(num_interactions, num_dims)
    pad_width = ((1, 1), (0, 0))
    At = jnp.einsum(
        "ndk,nd->nk", object_vectors, t, precision=jax.lax.Precision.HIGHEST
    )
    At = jnp.pad(At, pad_width, mode="constant", constant_values=0.0)
    Ap = jnp.einsum(
        "ndk,nd->nk", object_vectors, p, precision=jax.lax.Precision.HIGHEST
    )
    Ap = jnp.pad(Ap, pad_width, mode="constant", constant_values=0.0)
    b = jnp.concat((tx[None, :], object_origins, rx[None, :]), axis=-2)
    dAt = jnp.diff(At, axis=-2)
    dAp = jnp.diff(Ap, axis=-2)
    db = jnp.diff(b, axis=-2)
    dX = dAt + db
    num_1 = jnp.sum(dAp * dX, axis=-1)
    num_2 = jnp.sum(dAp * dAp, axis=-1)
    den = jnp.linalg.norm(dX + alpha * dAp, axis=-1)
    zero_den = den == 0.0

    num_1 = jnp.where(zero_den, 0.0, num_1)
    num_2 = jnp.where(zero_den, 0.0, num_2)
    den = jnp.where(zero_den, 1.0, den)

    left = jnp.sum(num_1 / den, axis=-1)
    right = jnp.sum(num_2 / den, axis=-1)
    zero_right = right == 0.0
    right = jnp.where(zero_right, 1.0, right)

    return jnp.where(zero_right, alpha, -left / right)


def _step_fn_general(
    state: _State,
    num_iters_linesearch: int,
    unroll_linesearch: int | bool,
) -> _State:
    p = -state.H @ state.grad
    init_alpha = jnp.array(1.0, dtype=p.dtype)
    alpha = jax.lax.scan(
        lambda a, _: (
            _fixed_point_fn_general(a, p, state.t, *state.params),
            None,
        ),
        init_alpha,
        xs=None,
        length=num_iters_linesearch,
        unroll=unroll_linesearch,
    )[0]
    s = alpha * p
    t = state.t + s

    grad = _grad_fn_general(t, *state.params)
    y = grad - state.grad

    Hy = jnp.matmul(state.H, y, precision=jax.lax.Precision.HIGHEST)
    yTHy = jnp.dot(y, Hy, precision=jax.lax.Precision.HIGHEST)
    sTy = jnp.dot(s, y, precision=jax.lax.Precision.HIGHEST)
    ssT = jnp.tensordot(s, s, axes=0, precision=jax.lax.Precision.HIGHEST)
    HysT = jnp.tensordot(Hy, s, axes=0, precision=jax.lax.Precision.HIGHEST)
    syTH = jnp.tensordot(s, Hy, axes=0, precision=jax.lax.Precision.HIGHEST)

    skip_update = sTy < jnp.finfo(sTy.dtype).eps
    sTy = jnp.where(skip_update, 1.0, sTy)
    H = jnp.where(
        skip_update,
        state.H,
        state.H + ((sTy + yTHy) * ssT) / (sTy**2) - (HysT + syTH) / sTy,
    )
    return _State(t, state.params, grad, H)


def _step_fn_im(
    state: _State,
    is_planar: jax.Array,
    n_hat: jax.Array,
    interaction_types: jax.Array,
    num_iters_linesearch: int,
    unroll_linesearch: int | bool,
) -> _State:
    tx, rx, object_origins, object_vectors = state.params
    p = -state.H @ state.grad
    p = jnp.where(is_planar, 0.0, p)

    xyz = _t_to_xyz_image_method(
        state.t,
        tx,
        rx,
        object_origins,
        object_vectors,
        is_planar,
        n_hat,
        interaction_types,
    )
    xyz_p = _t_to_xyz_image_method(
        state.t + p,
        tx,
        rx,
        object_origins,
        object_vectors,
        is_planar,
        n_hat,
        interaction_types,
    )
    dP = xyz_p - xyz
    full_X = jnp.concat([tx[None, :], xyz, rx[None, :]], axis=0)
    full_dP = jnp.concat(
        [
            jnp.zeros((1, 3), dtype=dP.dtype),
            dP,
            jnp.zeros((1, 3), dtype=dP.dtype),
        ],
        axis=0,
    )
    dX = jnp.diff(full_X, axis=0)
    dAp = jnp.diff(full_dP, axis=0)

    num_1 = jnp.sum(dAp * dX, axis=-1)
    num_2 = jnp.sum(dAp * dAp, axis=-1)

    def fp_linesearch(alpha: jax.Array, _: None) -> tuple[jax.Array, None]:
        den = jnp.linalg.norm(dX + alpha * dAp, axis=-1)
        zero_den = den == 0.0
        den = jnp.where(zero_den, 1.0, den)
        n1 = jnp.where(zero_den, 0.0, num_1)
        n2 = jnp.where(zero_den, 0.0, num_2)
        left = jnp.sum(n1 / den)
        right = jnp.sum(n2 / den)
        zero_r = right == 0.0
        safe_right = jnp.where(zero_r, 1.0, right)
        a_raw = -left / safe_right
        a_next = jnp.where(a_raw <= 0.0, 1.0, jnp.minimum(a_raw, 1.0))
        return jnp.where(zero_r, alpha, a_next), None

    init_alpha = jnp.array(1.0, dtype=p.dtype)
    ls_len = min(num_iters_linesearch, 5)
    alpha = jax.lax.scan(
        fp_linesearch,
        init_alpha,
        xs=None,
        length=ls_len,
        unroll=unroll_linesearch,
    )[0]

    s = alpha * p
    t = state.t + s

    xyz_next = _t_to_xyz_image_method(
        t,
        tx,
        rx,
        object_origins,
        object_vectors,
        is_planar,
        n_hat,
        interaction_types,
    )
    grad = _grad_fn_im(xyz_next, tx, rx, object_vectors, is_planar)
    y = grad - state.grad

    s = jnp.where(is_planar, 0.0, s)
    y = jnp.where(is_planar, 0.0, y)
    Hy = jnp.matmul(state.H, y, precision=jax.lax.Precision.HIGHEST)
    yTHy = jnp.dot(y, Hy, precision=jax.lax.Precision.HIGHEST)
    sTy = jnp.dot(s, y, precision=jax.lax.Precision.HIGHEST)
    ssT = jnp.tensordot(s, s, axes=0, precision=jax.lax.Precision.HIGHEST)
    HysT = jnp.tensordot(Hy, s, axes=0, precision=jax.lax.Precision.HIGHEST)
    syTH = jnp.tensordot(s, Hy, axes=0, precision=jax.lax.Precision.HIGHEST)

    skip_update = sTy < jnp.finfo(sTy.dtype).eps
    sTy = jnp.where(skip_update, 1.0, sTy)
    H = jnp.where(
        skip_update,
        state.H,
        state.H + ((sTy + yTHy) * ssT) / (sTy**2) - (HysT + syTH) / sTy,
    )
    return _State(t, state.params, grad, H)


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9))
def _trace_ray_t(
    tx: jax.Array,
    rx: jax.Array,
    object_origins: jax.Array,
    object_vectors: jax.Array,
    interaction_types: jax.Array,
    num_iters: int,
    unroll: int | bool,
    num_iters_linesearch: int,
    unroll_linesearch: int | bool,
    use_image_method: bool,
) -> jax.Array:
    num_interactions, num_dims, _ = object_vectors.shape
    dtype = jnp.result_type(tx, rx, object_origins, object_vectors)
    params = (tx, rx, object_origins, object_vectors)

    if use_image_method:
        is_planar, n_hat = _planar_info(object_vectors)
        has_diffraction = jnp.any(~is_planar) if num_dims > 0 else False
        n = num_interactions
        t = jnp.zeros(n, dtype=dtype)
        xyz0 = _t_to_xyz_image_method(
            t,
            tx,
            rx,
            object_origins,
            object_vectors,
            is_planar,
            n_hat,
            interaction_types,
        )
        if num_iters == 0 or num_dims == 0:
            return t

        grad = _grad_fn_im(xyz0, tx, rx, object_vectors, is_planar)
        H = jnp.identity(n, dtype=dtype)
        initial_state = _State(t, params, grad, H)

        final_state = jax.lax.scan(
            lambda state, _: (
                _step_fn_im(
                    state,
                    is_planar,
                    n_hat,
                    interaction_types,
                    num_iters_linesearch,
                    unroll_linesearch,
                ),
                None,
            ),
            initial_state,
            xs=None,
            length=num_iters,
            unroll=unroll,
        )[0]
        return jnp.where(has_diffraction, final_state.t, t)
    else:
        n = num_interactions * num_dims
        t = jnp.zeros(n, dtype=dtype)
        if num_dims == 0 or num_iters == 0:
            return t
        grad = jnp.zeros(n, dtype=dtype)
        H = jnp.identity(n, dtype=dtype)
        initial_state = _State(t, params, grad, H)
        final_state = jax.lax.scan(
            lambda state, _: (
                _step_fn_general(
                    state,
                    num_iters_linesearch,
                    unroll_linesearch,
                ),
                None,
            ),
            initial_state,
            xs=None,
            length=num_iters,
            unroll=unroll,
        )[0]
        return final_state.t


def _trace_ray_t_fwd(
    tx: jax.Array,
    rx: jax.Array,
    object_origins: jax.Array,
    object_vectors: jax.Array,
    interaction_types: jax.Array,
    num_iters: int,
    unroll: int | bool,
    num_iters_linesearch: int,
    unroll_linesearch: int | bool,
    use_image_method: bool,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
    t = _trace_ray_t(
        tx,
        rx,
        object_origins,
        object_vectors,
        interaction_types,
        num_iters,
        unroll,
        num_iters_linesearch,
        unroll_linesearch,
        use_image_method,
    )
    return t, (t, tx, rx, object_origins, object_vectors, interaction_types)


def _trace_ray_t_bwd(
    num_iters: int,
    unroll: int | bool,
    num_iters_linesearch: int,
    unroll_linesearch: int | bool,
    use_image_method: bool,
    res: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    cotangent: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, None]:
    del (
        num_iters,
        unroll,
        num_iters_linesearch,
        unroll_linesearch,
    )
    t, tx, rx, object_origins, object_vectors, interaction_types = res

    if use_image_method:
        is_planar, n_hat = _planar_info(object_vectors)

        def fun_t(t: jax.Array) -> jax.Array:
            xyz = _t_to_xyz_image_method(
                t,
                tx,
                rx,
                object_origins,
                object_vectors,
                is_planar,
                n_hat,
                interaction_types,
            )
            return _grad_fn_im(xyz, tx, rx, object_vectors, is_planar)

        _, vjp_fun_t = jax.vjp(fun_t, t)

        def matvec(u: jax.Array) -> jax.Array:
            return vjp_fun_t(u)[0]

        v = -cotangent
        A = jax.jacfwd(matvec)(jnp.zeros_like(v))
        reg_eye = jnp.diag(jnp.where(is_planar, 1.0, 0.0))
        u = jax.scipy.linalg.solve(A + reg_eye, v, assume_a="pos")

        def fun_args(
            tx_arg: jax.Array,
            rx_arg: jax.Array,
            origins_arg: jax.Array,
            vectors_arg: jax.Array,
        ) -> jax.Array:
            is_pl, nh = _planar_info(vectors_arg)
            xyz = _t_to_xyz_image_method(
                t,
                tx_arg,
                rx_arg,
                origins_arg,
                vectors_arg,
                is_pl,
                nh,
                interaction_types,
            )
            return _grad_fn_im(xyz, tx_arg, rx_arg, vectors_arg, is_pl)

        _, vjp_fun_args = jax.vjp(
            fun_args, tx, rx, object_origins, object_vectors
        )
        return (*vjp_fun_args(u), None)
    else:

        def fun_t_gen(t: jax.Array) -> jax.Array:
            return _grad_fn_general(t, tx, rx, object_origins, object_vectors)

        _, vjp_fun_t_gen = jax.vjp(fun_t_gen, t)

        def matvec_gen(u: jax.Array) -> jax.Array:
            return vjp_fun_t_gen(u)[0]

        v = -cotangent
        A = jax.jacfwd(matvec_gen)(jnp.zeros_like(v))
        u = jax.scipy.linalg.solve(A, v, assume_a="pos")

        def fun_args_gen(*args: jax.Array) -> jax.Array:
            return _grad_fn_general(t, *args)

        _, vjp_fun_args_gen = jax.vjp(
            fun_args_gen, tx, rx, object_origins, object_vectors
        )
        return (*vjp_fun_args_gen(u), None)


_trace_ray_t.defvjp(_trace_ray_t_fwd, _trace_ray_t_bwd)


def _trace_ray(
    tx: jax.Array,
    rx: jax.Array,
    object_origins: jax.Array,
    object_vectors: jax.Array,
    interaction_types: jax.Array,
    num_iters: int,
    unroll: int | bool,
    num_iters_linesearch: int,
    unroll_linesearch: int | bool,
    implicit_diff: bool,
    use_image_method: bool,
) -> jax.Array:
    if not implicit_diff:
        fun = _trace_ray_t.fun
    else:
        fun = _trace_ray_t

    t = fun(
        tx,
        rx,
        object_origins,
        object_vectors,
        interaction_types,
        num_iters=num_iters,
        unroll=unroll,
        num_iters_linesearch=num_iters_linesearch,
        unroll_linesearch=unroll_linesearch,
        use_image_method=use_image_method,
    )
    if use_image_method:
        is_planar, n_hat = _planar_info(object_vectors)
        return _t_to_xyz_image_method(
            t,
            tx,
            rx,
            object_origins,
            object_vectors,
            is_planar,
            n_hat,
            interaction_types,
        )
    else:
        return _t_to_xyz_general(t, object_origins, object_vectors)


## Public API ##


@partial(
    jax.jit,
    static_argnames=(
        "num_iters",
        "unroll",
        "num_iters_linesearch",
        "unroll_linesearch",
        "implicit_diff",
        "use_image_method",
    ),
)
def trace_rays(
    tx: jax.Array,
    rx: jax.Array,
    object_origins: jax.Array,
    object_vectors: jax.Array,
    interaction_types: jax.Array | None = None,
    *,
    num_iters: int = 100,
    unroll: int | bool = 1,
    num_iters_linesearch: int = 1,
    unroll_linesearch: int | bool = 1,
    implicit_diff: bool = True,
    use_image_method: bool = True,
) -> jax.Array:
    """Compute the points of interaction of rays with objects using Fermat's principle.

    Each ray is obtained by minimizing the total travel distance from transmitter to receiver
    using a quasi-Newton optimization algorithm (BFGS). When `use_image_method=True` (default),
    intermediate specular reflections and refractions/transmissions are solved in closed form
    via the exact Image Method, drastically reducing the optimization dimension to only the
    diffraction edge parameters.

    This function accepts batched inputs, where the leading dimensions must be broadcast-compatible.

    Args:
        tx: Transmitter positions of shape `(..., 3)`.
        rx: Receiver positions of shape `(..., 3)`.
        object_origins: Origins of the objects of shape `(..., num_interactions, 3)`.
        object_vectors: Vectors defining the objects of shape `(..., num_interactions, num_dims, 3)`.
        interaction_types: Optional interaction types of shape `(..., num_interactions)`
            (0 = reflection, 1 = diffraction, 2 = transmission). If omitted, planar surfaces
            are treated as specular reflections.
        num_iters: Number of iterations for the optimization algorithm (only needed for diffraction).
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
        use_image_method: If `True` (default), specular planar interactions are solved exactly in closed
            form using the Image Method, reducing optimization to only diffracting edges. If `False`,
            runs standard BFGS over all coordinates simultaneously.

    Returns:
        The points of interaction of shape `(..., num_interactions, 3)`.
        To include the transmitter and receiver positions, concatenate `tx` and `rx` to the result.
    """
    if interaction_types is None:
        num_interactions = object_origins.shape[-2]
        interaction_types = jnp.zeros(
            (*object_origins.shape[:-1],), dtype=jnp.int32
        )

    return jnp.vectorize(
        partial(
            _trace_ray,
            num_iters=num_iters,
            unroll=unroll,
            num_iters_linesearch=num_iters_linesearch,
            unroll_linesearch=unroll_linesearch,
            implicit_diff=implicit_diff,
            use_image_method=use_image_method,
        ),
        signature="(3),(3),(n,3),(n,d,3),(n)->(n,3)",
    )(tx, rx, object_origins, object_vectors, interaction_types)
