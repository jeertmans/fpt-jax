"""Validation utilities for checking ray paths against physical interaction laws.

Each interaction type is checked according to its fundamental optical law:
- **Reflection (`0` / `'reflection'`)**: Angle of incidence equals angle of
  reflection with respect to the surface normal:
  `cos(theta_in) == cos(theta_out)` and `sin(theta_in) == sin(theta_out)`.
- **Diffraction (`1` / `'diffraction'`)**: Angle of incidence equals angle of
  diffraction with respect to the edge direction (Keller's law of diffraction):
  `cos(theta_in) == cos(theta_out)` and `sin(theta_in) == sin(theta_out)`.
- **Transmission (`2` / `'transmission'`)**: Angle of incidence w.r.t. the
  incident normal equals angle of refraction/transmission w.r.t. the opposite
  normal:
  `cos(theta_in) == cos(theta_out)` and `sin(theta_in) == sin(theta_out)`.
"""

from enum import IntEnum
from typing import Any, Sequence

import numpy as np


class InteractionType(IntEnum):
    REFLECTION = 0
    DIFFRACTION = 1
    TRANSMISSION = 2


_TYPE_ALIASES: dict[str | int | InteractionType, InteractionType] = {
    InteractionType.REFLECTION: InteractionType.REFLECTION,
    InteractionType.DIFFRACTION: InteractionType.DIFFRACTION,
    InteractionType.TRANSMISSION: InteractionType.TRANSMISSION,
    0: InteractionType.REFLECTION,
    1: InteractionType.DIFFRACTION,
    2: InteractionType.TRANSMISSION,
    "reflection": InteractionType.REFLECTION,
    "specular": InteractionType.REFLECTION,
    "diffraction": InteractionType.DIFFRACTION,
    "transmission": InteractionType.TRANSMISSION,
    "refraction": InteractionType.TRANSMISSION,
}


def normalize_interaction_type(itype: str | int | InteractionType) -> InteractionType:
    if isinstance(itype, str):
        itype_clean = itype.strip().lower()
        if itype_clean in _TYPE_ALIASES:
            return _TYPE_ALIASES[itype_clean]
    elif itype in _TYPE_ALIASES:
        return _TYPE_ALIASES[itype]
    raise ValueError(
        f"Unknown interaction type: {itype!r}. Expected one of {list(_TYPE_ALIASES.keys())}"
    )


def check_path_interactions(
    tx: Any,
    rx: Any,
    path: Any,
    object_origins: Any,
    object_vectors: Any,
    interaction_list: Sequence[Any],
    *,
    atol: float = 1e-3,
    check_on_surface: bool = True,
    raise_on_error: bool = False,
) -> bool:
    """Validate that a ray path satisfies physical interaction laws.

    For each interaction point in `path`:
    - **Reflection (0)**: Angle of incidence equals angle of reflection w.r.t.
      the surface normal.
    - **Diffraction (1)**: Angle of incidence equals angle of diffraction
      w.r.t. the edge direction.
    - **Transmission (2)**: Angle of incidence w.r.t. the front normal equals
      angle of transmission w.r.t. the opposite normal.

    Angles are compared by verifying that both the dot product (cosine) and
    cross product norm (sine) with the reference direction evaluate to the same
    value within absolute tolerance `atol`.

    Args:
        tx: Transmitter position of shape `(3,)`.
        rx: Receiver position of shape `(3,)`.
        path: Sequence of interaction points of shape `(num_interactions, 3)`.
        object_origins: Origins of the objects of shape `(num_interactions, 3)`.
        object_vectors: Vectors defining the objects of shape
            `(num_interactions, num_dims, 3)`.
        interaction_list: List/array of interaction types of length
            `num_interactions` (0 = reflection, 1 = diffraction, 2 = transmission).
        atol: Absolute tolerance for angle comparisons and surface constraints.
        check_on_surface: Whether to verify that each interaction vertex lies on
            the object's plane or edge.
        raise_on_error: If True, raises an `AssertionError` describing the first
            violation instead of returning False.

    Returns:
        True if all interaction checks pass within `atol`, False otherwise.
    """
    tx_arr = np.asarray(tx, dtype=float)
    rx_arr = np.asarray(rx, dtype=float)
    path_arr = np.asarray(path, dtype=float)
    origins_arr = np.asarray(object_origins, dtype=float)
    vectors_arr = np.asarray(object_vectors, dtype=float)

    num_interactions = len(path_arr)
    if num_interactions == 0:
        return True

    if len(interaction_list) != num_interactions:
        msg = (
            f"Length of interaction_list ({len(interaction_list)}) does not "
            f"match path length ({num_interactions})"
        )
        if raise_on_error:
            raise AssertionError(msg)
        return False

    points = np.concatenate([tx_arr[None, :], path_arr, rx_arr[None, :]], axis=0)

    for i in range(num_interactions):
        itype = normalize_interaction_type(interaction_list[i])
        p_prev = points[i]
        p_curr = points[i + 1]
        p_next = points[i + 2]

        v_in = p_prev - p_curr
        norm_in = np.linalg.norm(v_in)
        if norm_in < 1e-12:
            if raise_on_error:
                raise AssertionError(
                    f"Interaction {i}: segment from previous point to vertex has near-zero length ({norm_in:.2e})"
                )
            return False
        v_in = v_in / norm_in

        v_out = p_next - p_curr
        norm_out = np.linalg.norm(v_out)
        if norm_out < 1e-12:
            if raise_on_error:
                raise AssertionError(
                    f"Interaction {i}: segment from vertex to next point has near-zero length ({norm_out:.2e})"
                )
            return False
        v_out = v_out / norm_out

        d_in = -v_in
        d_out = v_out

        vecs = vectors_arr[i]
        orig = origins_arr[i]
        r = p_curr - orig

        if itype == InteractionType.REFLECTION:
            u1, u2 = vecs[0], vecs[1]
            n = np.cross(u1, u2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-12:
                if raise_on_error:
                    raise AssertionError(
                        f"Interaction {i} (reflection): plane spanning vectors are collinear or zero"
                    )
                return False
            n = n / norm_n

            if check_on_surface:
                dist = abs(float(np.dot(r, n)))
                if dist > atol:
                    if raise_on_error:
                        raise AssertionError(
                            f"Interaction {i} (reflection): point {p_curr.tolist()} is not on the object plane (distance {dist:.2e} > atol {atol:.2e})"
                        )
                    return False

            # Normal pointing towards the incident ray side
            if np.dot(v_in, n) < 0:
                n = -n

            cos_in = float(np.dot(v_in, n))
            sin_in = float(np.linalg.norm(np.cross(v_in, n)))
            cos_out = float(np.dot(v_out, n))
            sin_out = float(np.linalg.norm(np.cross(v_out, n)))

            diff_cos = abs(cos_in - cos_out)
            diff_sin = abs(sin_in - sin_out)
            if diff_cos > atol or diff_sin > atol:
                if raise_on_error:
                    raise AssertionError(
                        f"Interaction {i} (reflection): angle mismatch w.r.t. surface normal. "
                        f"cos_in={cos_in:.6f}, cos_out={cos_out:.6f} (diff={diff_cos:.2e}), "
                        f"sin_in={sin_in:.6f}, sin_out={sin_out:.6f} (diff={diff_sin:.2e}), atol={atol:.2e}"
                    )
                return False

        elif itype == InteractionType.DIFFRACTION:
            e = vecs[0]
            norm_e = np.linalg.norm(e)
            if norm_e < 1e-12:
                if raise_on_error:
                    raise AssertionError(
                        f"Interaction {i} (diffraction): edge vector has near-zero length"
                    )
                return False
            e = e / norm_e

            if check_on_surface:
                r_perp = r - np.dot(r, e) * e
                dist = float(np.linalg.norm(r_perp))
                if dist > atol:
                    if raise_on_error:
                        raise AssertionError(
                            f"Interaction {i} (diffraction): point {p_curr.tolist()} is not on the edge line (distance {dist:.2e} > atol {atol:.2e})"
                        )
                    return False

            cos_in = float(np.dot(d_in, e))
            sin_in = float(np.linalg.norm(np.cross(d_in, e)))
            cos_out = float(np.dot(d_out, e))
            sin_out = float(np.linalg.norm(np.cross(d_out, e)))

            diff_cos = abs(cos_in - cos_out)
            diff_sin = abs(sin_in - sin_out)
            if diff_cos > atol or diff_sin > atol:
                if raise_on_error:
                    raise AssertionError(
                        f"Interaction {i} (diffraction): angle mismatch w.r.t. edge direction. "
                        f"cos_in={cos_in:.6f}, cos_out={cos_out:.6f} (diff={diff_cos:.2e}), "
                        f"sin_in={sin_in:.6f}, sin_out={sin_out:.6f} (diff={diff_sin:.2e}), atol={atol:.2e}"
                    )
                return False

        elif itype == InteractionType.TRANSMISSION:
            u1, u2 = vecs[0], vecs[1]
            n = np.cross(u1, u2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-12:
                if raise_on_error:
                    raise AssertionError(
                        f"Interaction {i} (transmission): plane spanning vectors are collinear or zero"
                    )
                return False
            n = n / norm_n

            if check_on_surface:
                dist = abs(float(np.dot(r, n)))
                if dist > atol:
                    if raise_on_error:
                        raise AssertionError(
                            f"Interaction {i} (transmission): point {p_curr.tolist()} is not on the object plane (distance {dist:.2e} > atol {atol:.2e})"
                        )
                    return False

            # Normal pointing towards the incident ray side
            if np.dot(v_in, n) < 0:
                n = -n

            n_trans = -n
            cos_in = float(np.dot(v_in, n))
            sin_in = float(np.linalg.norm(np.cross(v_in, n)))
            cos_out = float(np.dot(v_out, n_trans))
            sin_out = float(np.linalg.norm(np.cross(v_out, n_trans)))

            diff_cos = abs(cos_in - cos_out)
            diff_sin = abs(sin_in - sin_out)
            if diff_cos > atol or diff_sin > atol:
                if raise_on_error:
                    raise AssertionError(
                        f"Interaction {i} (transmission): angle mismatch w.r.t. interface normal. "
                        f"cos_in={cos_in:.6f}, cos_out={cos_out:.6f} (diff={diff_cos:.2e}), "
                        f"sin_in={sin_in:.6f}, sin_out={sin_out:.6f} (diff={diff_sin:.2e}), atol={atol:.2e}"
                    )
                return False

    return True


def assert_path_valid(
    tx: Any,
    rx: Any,
    path: Any,
    object_origins: Any,
    object_vectors: Any,
    interaction_list: Sequence[Any],
    *,
    atol: float = 1e-3,
    check_on_surface: bool = True,
) -> None:
    """Assert that a ray path satisfies physical interaction laws, raising AssertionError on failure."""
    check_path_interactions(
        tx,
        rx,
        path,
        object_origins,
        object_vectors,
        interaction_list,
        atol=atol,
        check_on_surface=check_on_surface,
        raise_on_error=True,
    )
