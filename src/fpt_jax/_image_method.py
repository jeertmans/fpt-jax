import jax
import jax.numpy as jnp


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
    del num_iters, num_iters_linesearch, unroll_linesearch, implicit_diff
    if object_vectors.shape[-2:] != (2, 3):
        raise ValueError("Expected object_vectors to have shape (..., 2, 3)")

    def image_of_vertex_with_respect_to_plane(
        vertex: jax.Array,
        object_origin: jax.Array,
        plane_normal: jax.Array,
    ) -> jax.Array:
        to_plane = vertex - object_origin
        return vertex - 2 * jnp.sum(to_plane * plane_normal) * plane_normal

    def intersection_of_ray_with_plane(
        ray_origin: jax.Array,
        ray_direction: jax.Array,
        object_origin: jax.Array,
        plane_normal: jax.Array,
    ) -> jax.Array:
        denom = jnp.sum(ray_direction * plane_normal)
        numer = jnp.sum((object_origin - ray_origin) * plane_normal)
        t = numer / denom
        return ray_origin + t * ray_direction

    def forward(
        previous_image: jax.Array,
        object_origin_and_normal: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        object_origin, plane_normal = object_origin_and_normal
        image = image_of_vertex_with_respect_to_plane(
            previous_image,
            object_origin,
            plane_normal,
        )
        return image, image

    def backward(
        previous_intersection: jax.Array,
        object_origin_and_normal_and_image: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        object_origin, plane_normal, image = object_origin_and_normal_and_image

        intersection = intersection_of_ray_with_plane(
            previous_intersection,
            image - previous_intersection,
            object_origin,
            plane_normal,
        )
        return intersection, intersection

    plane_normals = jnp.cross(object_vectors[:, 0, :], object_vectors[:, 1, :])

    _, images = jax.lax.scan(
        forward,
        init=tx,
        xs=(object_origins, plane_normals),
        unroll=unroll,
    )
    _, interaction_points = jax.lax.scan(
        backward,
        init=rx,
        xs=(object_origins, plane_normals, images),
        reverse=True,
        unroll=unroll,
    )

    return interaction_points


def valid_image_paths(
    object_origins: jax.Array,
    object_vectors: jax.Array,
    image_paths: jax.Array,
) -> jax.Array:
    d_prev = image_paths[..., :-2, :] - object_origins
    d_next = image_paths[..., +2:, :] - object_origins

    plane_normals = jnp.cross(object_vectors[:, 0, :], object_vectors[:, 1, :])

    dot_prev = jnp.sum(d_prev * plane_normals, axis=-1)
    dot_next = jnp.sum(d_next * plane_normals, axis=-1)

    return jnp.sign(dot_prev) == jnp.sign(dot_next)
