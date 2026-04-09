import jax
import jax.numpy as jnp


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


def length(
    tx: jax.Array,
    rx: jax.Array,
    xyz: jax.Array,
) -> jax.Array:
    return (
        jnp.linalg.norm(xyz[+0, :] - tx, axis=-1)
        + jnp.linalg.norm(xyz[+1:, :] - xyz[:-1, :], axis=-1).sum(axis=-1)
        + jnp.linalg.norm(rx - xyz[-1, :], axis=-1)
    )


def objective(
    t: jax.Array,
    tx: jax.Array,
    rx: jax.Array,
    object_origins: jax.Array,
    object_vectors: jax.Array,
) -> jax.Array:
    xyz = t_to_xyz(t, object_origins, object_vectors)
    return length(tx, rx, xyz)
