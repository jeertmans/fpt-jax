import re
from functools import partial
from typing import Literal
import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
import optimistix
from pytest_subtests import SubTests

from fpt_jax import trace_rays


def simple_diffraction_cases() -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
]:
    tx = jnp.array(
        [
            [+0.0, 0.0, +4.0],
            [+0.0, 0.0, +4.0],
            [+0.0, 0.0, +4.0],
            [+0.0, 0.0, +4.0],
        ]
    )
    rx = jnp.array(
        [
            [+4.0, 0.0, +4.0],
            [+4.0, +4.0, 0.0],
            [+4.0, 0.0, -4.0],
            [+4.0, -6.0, 0.0],
        ]
    )
    object_origins = jnp.array(
        [
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
        ]
    )
    object_vectors = jnp.array(
        [
            [[[1.0, 0.0, 0.0]]],
            [[[1.0, 0.0, 0.0]]],
            [[[1.0, 0.0, 0.0]]],
            [[[1.0, 0.0, 0.0]]],
        ]
    )

    expected = jnp.array(
        [
            [[+2.0, 0.0, 0.0]],
            [[+2.0, 0.0, 0.0]],
            [[+2.0, 0.0, 0.0]],
            [[+1.6, 0.0, 0.0]],
        ]
    )

    return tx, rx, object_origins, object_vectors, expected


def simple_reflection_cases(
    *, include_refraction_cases: bool = True
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    # Note: 'refraction' cases can be solved using path-length minimization, but not with the image method.
    tx = jnp.array(
        [
            [+3.0, 0.0, +4.0],
            [+0.0, 0.0, +4.0],
            [+0.0, 0.0, +4.0],
            [+0.0, 0.0, +4.0],
        ]
    )
    rx = jnp.array(
        [
            [+4.0, 0.0, +4.0],  # Possible
            [+4.0, 0.0, +4.0],  # Possible
            [+4.0, 0.0, -4.0],  # Impossible (= refraction)
            [+4.0, 0.0, -6.0],  # Impossible (= refraction)
        ]
    )
    object_origins = jnp.array(
        [
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
        ]
    )
    object_vectors = jnp.array(
        [
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        ]
    )

    expected = jnp.array(
        [
            [[+3.5, 0.0, 0.0]],
            [[+2.0, 0.0, 0.0]],
            [[+2.0, 0.0, 0.0]],
            [[+1.6, 0.0, 0.0]],
        ]
    )
    if not include_refraction_cases:
        return tx[:2], rx[:2], object_origins[:2], object_vectors[:2], expected[:2]
    return tx, rx, object_origins, object_vectors, expected


@pytest.mark.parametrize("cases", ["diffraction", "reflection"])
@pytest.mark.parametrize(
    "batched", [pytest.param(False, id="unbatched"), pytest.param(True, id="batched")]
)
@pytest.mark.parametrize(
    "num_iters", [pytest.param(4, id="4iters"), pytest.param(10, id="10iters")]
)
@pytest.mark.parametrize(
    "unroll", [pytest.param(False, id="loop"), pytest.param(True, id="unroll")]
)
@pytest.mark.parametrize(
    "num_iters_linesearch",
    [pytest.param(1, id="1iter_ls"), pytest.param(10, id="10iters_ls")],
)
@pytest.mark.parametrize(
    "unroll_linesearch",
    [pytest.param(False, id="loop_ls"), pytest.param(True, id="unroll_ls")],
)
@pytest.mark.parametrize(
    "insert_zeros",
    [pytest.param(False, id="2d"), pytest.param(True, id="2d+extra_dim")],
)
def test_trace_rays_simple_cases(
    cases: Literal["diffraction", "reflection"],
    batched: bool,
    num_iters: int,
    unroll: bool,
    num_iters_linesearch: int,
    unroll_linesearch: bool,
    insert_zeros: bool,
    subtests: SubTests,
):
    if cases == "diffraction":
        tx, rx, object_origins, object_vectors, expected = simple_diffraction_cases()
    else:  # cases == "reflection"
        tx, rx, object_origins, object_vectors, expected = simple_reflection_cases()

    if insert_zeros:  # insert extra dimension
        object_vectors = jnp.pad(
            object_vectors,
            ((0, 0), (0, 0), (0, 1), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )

    if not batched:
        for i in range(tx.shape[0]):
            with subtests.test(i=i):
                got = trace_rays(
                    tx[i],
                    rx[i],
                    object_origins[i],
                    object_vectors[i],
                    num_iters=num_iters,
                    unroll=unroll,
                    num_iters_linesearch=num_iters_linesearch,
                    unroll_linesearch=unroll_linesearch,
                )
                chex.assert_trees_all_close(got, expected[i])
    else:
        got = trace_rays(
            tx,
            rx,
            object_origins,
            object_vectors,
            num_iters=num_iters,
            unroll=unroll,
            num_iters_linesearch=num_iters_linesearch,
            unroll_linesearch=unroll_linesearch,
        )
        chex.assert_trees_all_close(got, expected)


@pytest.mark.parametrize(
    ("tx_shape", "objects_shape", "rx_shape", "expected_shape"),
    [
        ((), (2,), (), (2,)),
        ((), (5,), (), (5,)),
        ((4,), (5,), (), (4, 5)),
        ((4,), (5,), (1,), (4, 5)),
        (
            (4,),
            (
                4,
                5,
            ),
            (1,),
            (4, 5),
        ),
    ],
)
@pytest.mark.parametrize(
    "num_dims",
    [pytest.param(0, id="0d"), pytest.param(1, id="1d"), pytest.param(2, id="2d")],
)
def test_trace_rays_broadcasting_shapes(
    tx_shape: tuple[int, ...],
    objects_shape: tuple[int, ...],
    rx_shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
    num_dims: int,
):
    keys = jr.split(jr.PRNGKey(1234), 4)
    tx = jr.normal(keys[0], (*tx_shape, 3))
    object_origins = jr.normal(keys[1], (*objects_shape, 3))
    object_vectors = jr.normal(keys[2], (*objects_shape, num_dims, 3))
    rx = jr.normal(keys[3], (*rx_shape, 3))

    got = trace_rays(tx, rx, object_origins, object_vectors, num_iters=0)

    assert got.shape[:-1] == expected_shape


def t_to_xyz(
    t: jax.Array, object_origins: jax.Array, object_vectors: jax.Array
) -> jax.Array:
    *_, num_interactions, num_dims, _ = object_vectors.shape
    return object_origins + jnp.einsum(
        "...nd,...ndk->...nk",
        t.reshape(*t.shape[:-1], num_interactions, num_dims),
        object_vectors,
        precision=jax.lax.Precision.HIGHEST,
    )


def path_length(
    tx: jax.Array,
    rx: jax.Array,
    xyz: jax.Array,
) -> jax.Array:
    return (
        jnp.linalg.norm(xyz[+0, :] - tx, axis=-1)
        + jnp.linalg.norm(xyz[+1:, :] - xyz[:-1, :], axis=-1).sum(axis=-1)
        + jnp.linalg.norm(rx - xyz[-1, :], axis=-1)
    )


def objective_fn(
    t: jax.Array,
    tx: jax.Array,
    rx: jax.Array,
    object_origins: jax.Array,
    object_vectors: jax.Array,
) -> jax.Array:
    xyz = t_to_xyz(t, object_origins, object_vectors)
    return path_length(tx, rx, xyz)


@pytest.mark.parametrize(
    "num_diffractions",
    [
        pytest.param(1, id="N=1"),
        pytest.param(2, id="N=2"),
        pytest.param(3, id="N=3"),
        pytest.param(4, id="N=4"),
    ],
)
def test_trace_rays_zero_grad_at_solution(num_diffractions: int):
    keys = jr.split(jr.PRNGKey(1234), 4)
    batch = 1_000
    tx = jr.normal(keys[0], (batch, 3))
    edge_origins = jr.normal(keys[1], (batch, num_diffractions, 3))
    edge_vectors = jr.normal(keys[2], (batch, num_diffractions, 1, 3))
    rx = jr.normal(keys[3], (batch, 3))

    got = trace_rays(
        tx,
        rx,
        edge_origins,
        edge_vectors,
        num_iters=16,
        num_iters_linesearch=32,
    )

    # Got back to parametric variables
    got_t = (got[..., None, :] - edge_origins[..., None, :]) / edge_vectors
    got_t = jnp.max(got_t, axis=-1, where=edge_vectors != 0.0, initial=-jnp.inf)
    got_t = jnp.squeeze(got_t, axis=-1)
    grad = jax.vmap(jax.grad(objective_fn))(got_t, tx, rx, edge_origins, edge_vectors)
    norm_grad = jnp.linalg.norm(grad, axis=-1)

    chex.assert_trees_all_close(norm_grad, 0.0, atol=1e-3)


@partial(jnp.vectorize, signature="(3),(3),(n,3),(n,d,3)->(n,3)")
def trace_rays_with_optimistix(
    tx: jax.Array,
    rx: jax.Array,
    object_origins: jax.Array,
    object_vectors: jax.Array,
) -> jax.Array:
    num_iterations, num_dims, _ = object_vectors.shape
    n = num_iterations * num_dims
    dtype = jnp.result_type(tx, rx, object_origins, object_vectors)
    solver = optimistix.BFGS(atol=1e-16, rtol=1e-6)
    solution = optimistix.minimise(
        lambda t, args: objective_fn(t, *args),
        solver,
        y0=jnp.zeros(n, dtype=dtype),
        args=(tx, rx, object_origins, object_vectors),
        max_steps=512,
    )
    return t_to_xyz(solution.value, object_origins, object_vectors)


@pytest.mark.parametrize(
    "num_interactions", [pytest.param(1, id="N=1"), pytest.param(2, id="N=2")]
)
@pytest.mark.parametrize(
    "num_dims", [pytest.param(1, id="diffraction"), pytest.param(2, id="reflection")]
)
def test_trace_rays_vs_optimistix(num_interactions: int, num_dims: int):
    if num_interactions == 1 and num_dims == 2:
        pytest.skip("Convergence too difficult, resulting in inaccurate results.")

    keys = jr.split(jr.PRNGKey(1234), 4)
    batch = 100
    tx = jr.normal(keys[0], (batch, 3))
    object_origins = jr.normal(keys[1], (batch, num_interactions, 3))
    object_vectors = jr.normal(keys[2], (batch, num_interactions, num_dims, 3))
    rx = jr.normal(keys[3], (batch, 3))

    expected = trace_rays_with_optimistix(tx, rx, object_origins, object_vectors)

    got = trace_rays(
        tx,
        rx,
        object_origins,
        object_vectors,
        num_iters=1_000,
        num_iters_linesearch=1_000,
    )

    chex.assert_trees_all_close(got, expected, atol=1e-2)


@pytest.mark.parametrize("cases", ["diffraction", "reflection"])
@pytest.mark.parametrize(
    "num_iters", [pytest.param(4, id="4iters"), pytest.param(10, id="10iters")]
)
@pytest.mark.parametrize(
    "unroll", [pytest.param(False, id="loop"), pytest.param(True, id="unroll")]
)
@pytest.mark.parametrize(
    "num_iters_linesearch",
    [pytest.param(1, id="1iter_ls"), pytest.param(10, id="10iters_ls")],
)
@pytest.mark.parametrize(
    "unroll_linesearch",
    [pytest.param(False, id="loop_ls"), pytest.param(True, id="unroll_ls")],
)
def test_grad_trace_rays_simple_cases(
    cases: Literal["diffraction", "reflection"],
    num_iters: int,
    unroll: bool,
    num_iters_linesearch: int,
    unroll_linesearch: bool,
    subtests: SubTests,
):
    if cases == "diffraction":
        tx, rx, object_origins, object_vectors, expected = simple_diffraction_cases()
    else:  # cases == "reflection"
        tx, rx, object_origins, object_vectors, expected = simple_reflection_cases()

    def f(
        tx: jax.Array,
        rx: jax.Array,
        object_origins: jax.Array,
        object_vectors: jax.Array,
        implicit_diff: bool,
    ) -> jax.Array:
        xyz = trace_rays(
            tx,
            rx,
            object_origins,
            object_vectors,
            num_iters=num_iters,
            unroll=unroll,
            num_iters_linesearch=num_iters_linesearch,
            unroll_linesearch=unroll_linesearch,
            implicit_diff=implicit_diff,
        )
        return path_length(tx, rx, xyz)

    for i in range(tx.shape[0]):
        for arg_num in range(4):
            with subtests.test(i=i, arg_num=arg_num):
                expected = jax.grad(partial(f, implicit_diff=False), argnums=arg_num)(
                    tx[i],
                    rx[i],
                    object_origins[i],
                    object_vectors[i],
                )
                got = jax.grad(partial(f, implicit_diff=True), argnums=arg_num)(
                    tx[i],
                    rx[i],
                    object_origins[i],
                    object_vectors[i],
                )
                chex.assert_trees_all_close(got, expected, atol=1e-6)

                # Also check that forward-mode autodiff works when not using implicit differentiation
                _ = jax.jacfwd(partial(f, implicit_diff=False), argnums=arg_num)(
                    tx[i],
                    rx[i],
                    object_origins[i],
                    object_vectors[i],
                )

                # But not when using implicit differentiation
                with pytest.raises(
                    TypeError,
                    match=re.escape(
                        "can't apply forward-mode autodiff (jvp) to a custom_vjp function"
                    ),
                ):
                    _ = jax.jacfwd(partial(f, implicit_diff=True), argnums=arg_num)(
                        tx[i],
                        rx[i],
                        object_origins[i],
                        object_vectors[i],
                    )


@pytest.mark.parametrize(
    "num_iters", [pytest.param(4, id="4iters"), pytest.param(10, id="10iters")]
)
@pytest.mark.parametrize(
    "unroll", [pytest.param(False, id="loop"), pytest.param(True, id="unroll")]
)
@pytest.mark.parametrize(
    "num_iters_linesearch",
    [pytest.param(1, id="1iter_ls"), pytest.param(10, id="10iters_ls")],
)
@pytest.mark.parametrize(
    "unroll_linesearch",
    [pytest.param(False, id="loop_ls"), pytest.param(True, id="unroll_ls")],
)
def test_grad_trace_rays_simple_vs_image_method(
    num_iters: int,
    unroll: bool,
    num_iters_linesearch: int,
    unroll_linesearch: bool,
    subtests: SubTests,
):
    tx, rx, object_origins, object_vectors, expected = simple_reflection_cases(
        include_refraction_cases=False
    )

    def f(
        tx: jax.Array,
        rx: jax.Array,
        object_origins: jax.Array,
        object_vectors: jax.Array,
        use_image_method: bool,
    ) -> jax.Array:
        if use_image_method:
            xyz = trace_rays_with_image_method(tx, rx, object_origins, object_vectors)
        else:
            xyz = trace_rays(
                tx,
                rx,
                object_origins,
                object_vectors,
                num_iters=num_iters,
                unroll=unroll,
                num_iters_linesearch=num_iters_linesearch,
                unroll_linesearch=unroll_linesearch,
            )
        return path_length(tx, rx, xyz)

    for i in range(tx.shape[0]):
        for arg_num in range(4):
            with subtests.test(i=i, arg_num=arg_num):
                expected = jax.grad(
                    partial(f, use_image_method=False), argnums=arg_num
                )(
                    tx[i],
                    rx[i],
                    object_origins[i],
                    object_vectors[i],
                )
                got = jax.grad(partial(f, use_image_method=False), argnums=arg_num)(
                    tx[i],
                    rx[i],
                    object_origins[i],
                    object_vectors[i],
                )
                chex.assert_trees_all_close(got, expected)


@pytest.mark.parametrize(
    "num_interactions", [pytest.param(1, id="N=1"), pytest.param(2, id="N=2")]
)
@pytest.mark.parametrize(
    "num_dims", [pytest.param(1, id="diffraction"), pytest.param(2, id="reflection")]
)
def test_grad_trace_rays_vs_optimistix(
    num_interactions: int, num_dims: int, subtests: SubTests
):
    if num_interactions == 2 and num_dims == 2:
        pytest.skip("Convergence too difficult, resulting in inaccurate gradients.")

    keys = jr.split(jr.PRNGKey(1234), 4)
    batch = 100
    tx = jr.normal(keys[0], (batch, 3))
    object_origins = jr.normal(keys[1], (batch, num_interactions, 3))
    object_vectors = jr.normal(keys[2], (batch, num_interactions, num_dims, 3))
    rx = jr.normal(keys[3], (batch, 3))

    def f(
        tx: jax.Array,
        rx: jax.Array,
        object_origins: jax.Array,
        object_vectors: jax.Array,
        use_optimistix: bool,
    ) -> jax.Array:
        if use_optimistix:
            xyz = trace_rays_with_optimistix(tx, rx, object_origins, object_vectors)
        else:
            xyz = trace_rays(
                tx,
                rx,
                object_origins,
                object_vectors,
                num_iters=1_000,
                num_iters_linesearch=512,
            )
        return path_length(tx, rx, xyz)

    for arg_num in range(4):
        with subtests.test(arg_num=arg_num):
            expected = jax.vmap(
                jax.grad(partial(f, use_optimistix=True), argnums=arg_num)
            )(tx, rx, object_origins, object_vectors)
            got = jax.vmap(jax.grad(partial(f, use_optimistix=False), argnums=arg_num))(
                tx, rx, object_origins, object_vectors
            )

            chex.assert_trees_all_close(got, expected, atol=2e-2)


# Image method: code adapted from DiffeRT


@partial(jnp.vectorize, signature="(3),(3),(n,3),(n,d,3)->(n,3)")
def trace_rays_with_image_method(
    tx: jax.Array,
    rx: jax.Array,
    plane_origins: jax.Array,
    plane_vectors: jax.Array,
) -> jax.Array:
    def image_of_vertex_with_respect_to_plane(
        vertex: jax.Array,
        plane_origin: jax.Array,
        plane_normal: jax.Array,
    ) -> jax.Array:
        to_plane = vertex - plane_origin
        return vertex - 2 * jnp.sum(to_plane * plane_normal) * plane_normal

    def intersection_of_ray_with_plane(
        ray_origin: jax.Array,
        ray_direction: jax.Array,
        plane_origin: jax.Array,
        plane_normal: jax.Array,
    ) -> jax.Array:
        denom = jnp.sum(ray_direction * plane_normal)
        numer = jnp.sum((plane_origin - ray_origin) * plane_normal)
        t = numer / denom
        return ray_origin + t * ray_direction

    def forward(
        previous_image: jax.Array,
        plane_origin_and_normal: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        plane_origin, plane_normal = plane_origin_and_normal
        image = image_of_vertex_with_respect_to_plane(
            previous_image,
            plane_origin,
            plane_normal,
        )
        return image, image

    def backward(
        previous_intersection: jax.Array,
        plane_origin_and_normal_and_image: tuple[
            jax.Array,
            jax.Array,
            jax.Array,
        ],
    ) -> tuple[jax.Array, jax.Array]:
        plane_origin, plane_normal, image = plane_origin_and_normal_and_image

        intersection = intersection_of_ray_with_plane(
            previous_intersection,
            image - previous_intersection,
            plane_origin,
            plane_normal,
        )
        return intersection, intersection

    plane_normals = jnp.cross(plane_vectors[:, 0, :], plane_vectors[:, 1, :])

    _, images = jax.lax.scan(
        forward,
        init=tx,
        xs=(plane_origins, plane_normals),
    )
    _, interaction_points = jax.lax.scan(
        backward,
        init=rx,
        xs=(plane_origins, plane_normals, images),
        reverse=True,
    )

    return interaction_points


def valid_reflection_paths(
    plane_origins: jax.Array,
    plane_normals: jax.Array,
    ray_paths: jax.Array,
) -> jax.Array:
    all_finite = jnp.isfinite(ray_paths).all(axis=(-1, -2))

    d_prev = ray_paths[..., :-2, :] - plane_origins
    d_next = ray_paths[..., +2:, :] - plane_origins

    dot_prev = jnp.sum(d_prev * plane_normals, axis=-1)
    dot_next = jnp.sum(d_next * plane_normals, axis=-1)
    same_sign = (jnp.sign(dot_prev) == jnp.sign(dot_next)).all(axis=-1)
    return all_finite & same_sign


@pytest.mark.parametrize(
    "num_reflections",
    [
        pytest.param(1, id="N=1"),
        pytest.param(2, id="N=2"),
        pytest.param(3, id="N=3"),
        pytest.param(4, id="N=4"),
    ],
)
def test_trace_rays_vs_image_method(num_reflections: int):
    keys = jr.split(jr.PRNGKey(1234), 4)
    batch = 5
    tx = jr.normal(keys[0], (batch, 3))
    plane_origins = 100 * jr.normal(keys[1], (batch, num_reflections, 3))
    plane_vectors = jr.normal(keys[2], (batch, num_reflections, 2, 3))
    plane_vectors /= jnp.linalg.norm(plane_vectors, axis=-1, keepdims=True)
    plane_vectors = plane_vectors.at[..., 1, :].set(
        jnp.cross(plane_vectors[..., 0, :], plane_vectors[..., 1, :])
    )
    plane_vectors = plane_vectors.at[..., 1, :].set(
        plane_vectors[..., 1, :]
        / jnp.linalg.norm(plane_vectors[..., 1, :], axis=-1, keepdims=True)
    )
    plane_normals = jnp.cross(plane_vectors[..., 0, :], plane_vectors[..., 1, :])
    rx = jr.normal(keys[3], (batch, 3))

    chex.assert_trees_all_close(jnp.linalg.norm(plane_vectors, axis=-1), 1.0)
    chex.assert_trees_all_close(
        (plane_vectors[..., 0, :] * plane_vectors[..., 1, :]).sum(axis=-1),
        0.0,
        atol=1e-6,
    )
    chex.assert_trees_all_close(jnp.linalg.norm(plane_normals, axis=-1), 1.0)

    with jax.debug_nans(False):
        expected = trace_rays_with_image_method(tx, rx, plane_origins, plane_vectors)

    valid = valid_reflection_paths(
        plane_origins,
        plane_normals,
        jnp.concat([tx[:, None, :], expected, rx[:, None, :]], axis=1),
    )
    expected = jnp.where(valid[..., None, None], expected, 0.0)
    got = trace_rays(
        tx,
        rx,
        plane_origins,
        plane_vectors,
        num_iters=1_000,
        num_iters_linesearch=64,
    )
    got = jnp.where(valid[..., None, None], got, 0.0)

    chex.assert_trees_all_close(got, expected, atol=1e-3)
