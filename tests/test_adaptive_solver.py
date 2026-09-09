import pytest
import jax
import jax.numpy as jnp
import chex

import fpt_jax
from fpt_jax import trace_rays
from fpt_jax._core import _cauchy_termination, _max_norm


def test_private_exports():
    assert fpt_jax.__all__ == ("trace_rays",)
    assert not hasattr(fpt_jax, "cauchy_termination")
    assert not hasattr(fpt_jax, "max_norm")


def test_max_norm():
    assert float(_max_norm(jnp.array([]))) == 0.0
    x = jnp.array([-3.5, 2.0, 1.0, -1.2])
    assert float(_max_norm(x)) == 3.5
    mat = jnp.array([[-10.0, 5.0], [2.0, -8.0]])
    assert float(_max_norm(mat)) == 10.0


def test_cauchy_termination_unit():
    rtol = 1e-3
    atol = 1e-4

    # Large diff: should NOT terminate
    y = jnp.array([1.0, 2.0])
    y_diff = jnp.array([0.1, 0.05])
    f = jnp.array(10.0)
    f_diff = jnp.array(0.5)
    assert not bool(
        _cauchy_termination(
            rtol, atol, norm=_max_norm, y=y, y_diff=y_diff, f=f, f_diff=f_diff
        )
    )

    # Small diff: SHOULD terminate
    y_diff_small = jnp.array([1e-5, 1e-5])
    f_diff_small = jnp.array(1e-5)
    assert bool(
        _cauchy_termination(
            rtol,
            atol,
            norm=_max_norm,
            y=y,
            y_diff=y_diff_small,
            f=f,
            f_diff=f_diff_small,
        )
    )

    # Small y_diff but large f_diff: should NOT terminate
    assert not bool(
        _cauchy_termination(
            rtol, atol, norm=_max_norm, y=y, y_diff=y_diff_small, f=f, f_diff=f_diff
        )
    )

    # Positional args calling convention
    assert bool(
        _cauchy_termination(rtol, atol, _max_norm, y, y_diff_small, f, f_diff_small)
    )
    # Omitting norm calling convention
    assert bool(_cauchy_termination(rtol, atol, y, y_diff_small, f, f_diff_small))


def test_trace_rays_validation():
    tx = jnp.zeros(3)
    rx = jnp.array([1.0, 0.0, 0.0])
    origins = jnp.zeros((1, 3))
    vectors = jnp.zeros((1, 1, 3))

    # Neither num_iters nor max_num_iters
    with pytest.raises(
        ValueError, match="Exactly one of 'num_iters' or 'max_num_iters' must be set"
    ):
        trace_rays(tx, rx, origins, vectors)

    # Both num_iters and max_num_iters
    with pytest.raises(
        ValueError, match="Exactly one of 'num_iters' or 'max_num_iters' must be set"
    ):
        trace_rays(
            tx,
            rx,
            origins,
            vectors,
            num_iters=10,
            max_num_iters=20,
            rtol=1e-3,
            atol=1e-3,
        )

    # Negative num_iters
    with pytest.raises(ValueError, match="'num_iters' must be a non-negative integer"):
        trace_rays(tx, rx, origins, vectors, num_iters=-1)

    # Negative max_num_iters
    with pytest.raises(
        ValueError, match="'max_num_iters' must be a non-negative integer"
    ):
        trace_rays(tx, rx, origins, vectors, max_num_iters=-1, rtol=1e-3, atol=1e-3)

    # Missing rtol or atol
    with pytest.raises(ValueError, match="'rtol' and 'atol' are required"):
        trace_rays(tx, rx, origins, vectors, max_num_iters=10)
    with pytest.raises(ValueError, match="'rtol' and 'atol' are required"):
        trace_rays(tx, rx, origins, vectors, max_num_iters=10, rtol=1e-3)
    with pytest.raises(ValueError, match="'rtol' and 'atol' are required"):
        trace_rays(tx, rx, origins, vectors, max_num_iters=10, atol=1e-3)

    # Negative rtol or atol
    with pytest.raises(ValueError, match="'rtol' and 'atol' must be non-negative"):
        trace_rays(tx, rx, origins, vectors, max_num_iters=10, rtol=-1e-3, atol=1e-3)
    with pytest.raises(ValueError, match="'rtol' and 'atol' must be non-negative"):
        trace_rays(tx, rx, origins, vectors, max_num_iters=10, rtol=1e-3, atol=-1e-3)

    # Unroll != 1 and != False with max_num_iters
    with pytest.raises(ValueError, match="'unroll' must be 1 or False"):
        trace_rays(
            tx, rx, origins, vectors, max_num_iters=10, rtol=1e-3, atol=1e-3, unroll=2
        )
    with pytest.raises(ValueError, match="'unroll' must be 1 or False"):
        trace_rays(
            tx,
            rx,
            origins,
            vectors,
            max_num_iters=10,
            rtol=1e-3,
            atol=1e-3,
            unroll=True,
        )

    # implicit_diff=False with max_num_iters
    with pytest.raises(
        ValueError,
        match="Reverse-mode differentiation through a while loop is not supported",
    ):
        trace_rays(
            tx,
            rx,
            origins,
            vectors,
            max_num_iters=10,
            rtol=1e-3,
            atol=1e-3,
            implicit_diff=False,
        )


def test_adaptive_trace_rays_convergence():
    # 1 diffraction edge along Z axis: origin=(0, 0, -10), vector=(0, 0, 20)
    tx = jnp.array([1.0, 1.0, -2.0])
    rx = jnp.array([1.0, 1.0, 2.0])
    origins = jnp.array([[0.0, 0.0, -10.0]])
    vectors = jnp.array([[[0.0, 0.0, 20.0]]])
    itypes = jnp.array([1])

    # True Fermat point for symmetric tx and rx is at z=0 (t = 0.5) -> (0, 0, 0)
    expected = jnp.array([[0.0, 0.0, 0.0]])

    # Test with fixed iterations
    got_fixed = trace_rays(tx, rx, origins, vectors, itypes, num_iters=10)
    chex.assert_trees_all_close(got_fixed, expected, atol=1e-4)

    # Test with adaptive while_loop
    got_adaptive = trace_rays(
        tx,
        rx,
        origins,
        vectors,
        itypes,
        max_num_iters=50,
        rtol=1e-5,
        atol=1e-5,
        unroll=1,
    )
    chex.assert_trees_all_close(got_adaptive, expected, atol=1e-4)
    chex.assert_trees_all_close(got_adaptive, got_fixed, atol=1e-4)


def test_adaptive_trace_rays_gradient():
    # Verify implicit differentiation works through max_num_iters
    tx = jnp.array([1.0, 1.0, -2.0])
    rx = jnp.array([1.0, 1.0, 2.0])
    origins = jnp.array([[0.0, 0.0, -10.0]])
    vectors = jnp.array([[[0.0, 0.0, 20.0]]])
    itypes = jnp.array([1])

    def loss(tx_pos):
        pts = trace_rays(
            tx_pos,
            rx,
            origins,
            vectors,
            itypes,
            max_num_iters=50,
            rtol=1e-5,
            atol=1e-5,
            implicit_diff=True,
        )
        return jnp.sum(pts**2)

    grad = jax.grad(loss)(tx)
    assert jnp.all(jnp.isfinite(grad))
