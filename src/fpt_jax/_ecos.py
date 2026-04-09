"""SOCP solver for path-length minimization using a Primal-Dual Interior Point method.

Uses the arrow-matrix representation for SOC cones with Mehrotra predictor-corrector.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple

from ._common import t_to_xyz


class _State(NamedTuple):
    x: jax.Array  # primal decision variables (chi)
    s: jax.Array  # slack variables (in cone)
    z: jax.Array  # dual variables (in cone)


def _build_socp(
    tx: jax.Array,
    rx: jax.Array,
    object_origins: jax.Array,
    object_vectors: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build the SOCP matrices c, G, h for path-length minimization."""
    num_interactions, num_dims, d_space = object_vectors.shape
    n = num_interactions
    n_t = n * num_dims
    n_v = n + 1
    n_x = n_t + n_v
    cone_dim = 1 + d_space
    num_cones = n + 1
    m = num_cones * cone_dim

    dtype = jnp.result_type(tx, rx, object_origins, object_vectors)

    c = jnp.concatenate([jnp.zeros(n_t, dtype=dtype), jnp.ones(n_v, dtype=dtype)])

    b = jnp.concatenate([tx[None, :], object_origins, rx[None, :]], axis=0)

    G = jnp.zeros((m, n_x), dtype=dtype)
    h = jnp.zeros(m, dtype=dtype)

    for i in range(num_cones):
        row_start = i * cone_dim
        G = G.at[row_start, n_t + i].set(-1.0)
        h = h.at[row_start + 1 : row_start + 1 + d_space].set(b[i + 1] - b[i])

        if 1 <= i <= n:
            idx = i - 1
            t_start = idx * num_dims
            Ai = object_vectors[idx]
            G = G.at[
                row_start + 1 : row_start + 1 + d_space, t_start : t_start + num_dims
            ].add(Ai.T)

        if 1 <= i + 1 <= n:
            idx = i
            t_start = idx * num_dims
            Ai1 = object_vectors[idx]
            G = G.at[
                row_start + 1 : row_start + 1 + d_space, t_start : t_start + num_dims
            ].add(-Ai1.T)

    return c, G, h


def _soc_product(u: jax.Array, v: jax.Array, cone_dim: int) -> jax.Array:
    """Jordan product u ∘ v for product of SOCs."""
    nc = u.shape[0] // cone_dim
    ur = u.reshape(nc, cone_dim)
    vr = v.reshape(nc, cone_dim)
    r0 = jnp.sum(ur * vr, axis=1, keepdims=True)
    rb = ur[:, 0:1] * vr[:, 1:] + vr[:, 0:1] * ur[:, 1:]
    return jnp.concatenate([r0, rb], axis=1).ravel()


def _arw(x: jax.Array, cone_dim: int) -> jax.Array:
    """Block-diagonal arrow matrix for product of SOCs.

    arw(x_k) = [[x0, xbar^T], [xbar, x0*I]] for each cone k.
    """
    m = x.shape[0]
    nc = m // cone_dim
    xr = x.reshape(nc, cone_dim)

    W = jnp.zeros((m, m), dtype=x.dtype)
    offsets = jnp.arange(nc) * cone_dim

    for j in range(cone_dim):
        idx = offsets + j
        W = W.at[idx, idx].set(xr[:, 0])

    for j in range(1, cone_dim):
        r_idx = offsets
        c_idx = offsets + j
        W = W.at[r_idx, c_idx].set(xr[:, j])
        W = W.at[c_idx, r_idx].set(xr[:, j])

    return W


def _soc_max_step(x: jax.Array, dx: jax.Array, cone_dim: int) -> jax.Array:
    """Max alpha in (0, 1] s.t. x + alpha*dx stays in SOC for each cone."""
    nc = x.shape[0] // cone_dim
    xr = x.reshape(nc, cone_dim)
    dxr = dx.reshape(nc, cone_dim)

    x0 = xr[:, 0]
    xb = xr[:, 1:]
    dx0 = dxr[:, 0]
    dxb = dxr[:, 1:]

    alpha_lin = jnp.where(dx0 < 0, -x0 / dx0, jnp.inf)

    c_val = x0**2 - jnp.sum(xb**2, axis=1)
    b_val = 2.0 * (x0 * dx0 - jnp.sum(xb * dxb, axis=1))
    a_val = dx0**2 - jnp.sum(dxb**2, axis=1)

    disc = b_val**2 - 4.0 * a_val * c_val
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))

    is_quad = jnp.abs(a_val) > 1e-30
    denom_safe = jnp.where(is_quad, 2.0 * a_val, 1.0)

    r1 = (-b_val - sqrt_disc) / denom_safe
    r2 = (-b_val + sqrt_disc) / denom_safe
    rmin = jnp.minimum(r1, r2)
    rmax = jnp.maximum(r1, r2)

    alpha_lin_case = jnp.where(b_val < -1e-30, -c_val / b_val, jnp.inf)
    alpha_neg = jnp.where(rmax > 0, rmax, jnp.inf)
    alpha_pos = jnp.where((disc > 0) & (rmin > 0), rmin, jnp.inf)

    alpha_quad = jnp.where(
        is_quad,
        jnp.where(a_val < 0, alpha_neg, alpha_pos),
        alpha_lin_case,
    )

    alpha_per_cone = jnp.minimum(alpha_lin, alpha_quad)
    alpha_per_cone = jnp.maximum(alpha_per_cone, 0.0)
    return jnp.minimum(jnp.min(alpha_per_cone), 1.0)


def _step_fn(
    state: _State,
    num_iters_linesearch: int,
    unroll_linesearch: int | bool,
    c: jax.Array,
    G: jax.Array,
    h: jax.Array,
    num_cones: int,
    cone_dim: int,
) -> _State:
    """One Mehrotra predictor-corrector PDIP step for SOCP.

    The complementarity condition s ∘ dz + z ∘ ds = rhs is linearized as:
      arw(s) dz + arw(z) ds = rhs
    where arw(x) is the arrow matrix representation of x.

    After eliminating ds = -rp - G dx:
      G^T arw(s)^{-1} arw(z) G dx = -rd - G^T arw(s)^{-1}(rhs + arw(z) rp)

    Key identity: arw(s)^{-1}(s ∘ z) = arw(s)^{-1} arw(s) z = z
    So arw(s)^{-1}(-s ∘ z) = -z.
    """
    x, s, z = state
    m = s.shape[0]

    # Residuals
    rp = G @ x + s - h
    rd = c + G.T @ z
    mu = jnp.dot(s, z) / num_cones

    # Build arrow matrices and solve system
    Arw_s = _arw(s, cone_dim)
    Arw_z = _arw(z, cone_dim)

    # Regularize arw(s) for numerical stability
    Arw_s_reg = Arw_s + 1e-10 * jnp.eye(m, dtype=s.dtype)

    # arw(s)^{-1} arw(z)
    Arw_s_inv_z = jnp.linalg.solve(Arw_s_reg, Arw_z)

    # Normal equation matrix
    H = G.T @ Arw_s_inv_z @ G
    H = H + 1e-8 * jnp.eye(H.shape[0], dtype=H.dtype)

    # ─── Affine direction (sigma = 0) ───────────────────────────
    sz = _soc_product(s, z, cone_dim)

    # RHS: -rd - G^T arw(s)^{-1}(-sz + arw(z) rp)
    # = -rd - G^T(-z + arw(s)^{-1} arw(z) rp)  [since arw(s)^{-1}(-sz) = -z]
    # = -rd + G^T z - G^T Arw_s_inv_z rp
    rhs_aff = -rd + G.T @ z - G.T @ (Arw_s_inv_z @ rp)

    dx_aff = jnp.linalg.solve(H, rhs_aff)
    ds_aff = -rp - G @ dx_aff
    # dz = arw(s)^{-1}(-sz - arw(z) ds_aff)
    # = -z - Arw_s_inv_z @ ds_aff
    dz_aff = -z - Arw_s_inv_z @ ds_aff

    # Affine step size
    alpha_aff = jnp.minimum(
        _soc_max_step(s, ds_aff, cone_dim),
        _soc_max_step(z, dz_aff, cone_dim),
    )

    # Centering parameter (Mehrotra heuristic)
    mu_aff = jnp.dot(s + alpha_aff * ds_aff, z + alpha_aff * dz_aff) / num_cones
    sigma = jnp.where(mu > 1e-20, jnp.clip((mu_aff / mu) ** 3, 1e-4, 1.0), 1e-4)

    # ─── Combined direction ─────────────────────────────────────
    dsz_aff = _soc_product(ds_aff, dz_aff, cone_dim)
    e = jnp.zeros(m, dtype=s.dtype)
    e = e.reshape(num_cones, cone_dim).at[:, 0].set(1.0).ravel()

    # Complementarity RHS: -sz + sigma*mu*e - dsz_aff
    rhs_comp = -sz + sigma * mu * e - dsz_aff

    # arw(s)^{-1}(rhs_comp + arw(z) rp)
    sinv_rhs = jnp.linalg.solve(Arw_s_reg, rhs_comp)
    rhs_cc = -rd - G.T @ (sinv_rhs + Arw_s_inv_z @ rp)

    dx_cc = jnp.linalg.solve(H, rhs_cc)
    ds_cc = -rp - G @ dx_cc
    dz_cc = jnp.linalg.solve(Arw_s_reg, rhs_comp - Arw_z @ ds_cc)

    # Step size with safety margin
    alpha = 0.99 * jnp.minimum(
        _soc_max_step(s, ds_cc, cone_dim),
        _soc_max_step(z, dz_cc, cone_dim),
    )
    alpha = jnp.clip(alpha, 0.0, 1.0)

    x_new = x + alpha * dx_cc
    s_new = s + alpha * ds_cc
    z_new = z + alpha * dz_cc

    # NaN guard: keep old state if numerical issues
    has_nan = (
        jnp.any(jnp.isnan(x_new))
        | jnp.any(jnp.isnan(s_new))
        | jnp.any(jnp.isnan(z_new))
    )
    x_out = jnp.where(has_nan, x, x_new)
    s_out = jnp.where(has_nan, s, s_new)
    z_out = jnp.where(has_nan, z, z_new)

    return _State(x_out, s_out, z_out)


def _initialize_socp(
    c: jax.Array,
    G: jax.Array,
    h: jax.Array,
    num_cones: int,
    cone_dim: int,
) -> _State:
    """Initialize primal/dual in cone interiors."""
    n_x = c.shape[0]
    m = G.shape[0]
    dtype = c.dtype

    x = jnp.zeros(n_x, dtype=dtype)

    s0 = h - G @ x
    s_r = s0.reshape(num_cones, cone_dim)
    s0v = s_r[:, 0]
    sb = s_r[:, 1:]
    sb_norms = jnp.linalg.norm(sb, axis=1)
    new_s0 = jnp.maximum(s0v, sb_norms + 1.0)
    s = s_r.at[:, 0].set(new_s0).ravel()

    z = jnp.zeros(m, dtype=dtype)
    z = z.reshape(num_cones, cone_dim).at[:, 0].set(1.0).ravel()

    return _State(x, s, z)


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
    if n == 0:
        return t

    c, G, h = _build_socp(tx, rx, object_origins, object_vectors)
    num_cones = num_interactions + 1
    cone_dim = 1 + object_vectors.shape[2]

    initial_state = _initialize_socp(c, G, h, num_cones, cone_dim)

    final_state = jax.lax.scan(
        lambda state, _: (
            _step_fn(
                state,
                num_iters_linesearch,
                unroll_linesearch,
                c,
                G,
                h,
                num_cones,
                cone_dim,
            ),
            None,
        ),
        initial_state,
        xs=None,
        length=num_iters,
        unroll=unroll,
    )[0]

    return final_state.x[:n]


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
    del implicit_diff  # Not implemented yet
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
