"""Generate `tests/data/sionna_paths.toml` from Sionna RT city scenes.

This is a utility module, not a test module: it is run by `tests/conftest.py`
(in a `pytest_configure` hook, prior to test collection) when
`--generate-sionna-dataset` is passed to pytest, to (re)generate the dataset
file. It uses Sionna RT (https://nvlabs.github.io/sionna/rt/) to compute
realistic multi-bounce reflection/diffraction/transmission paths on built-in
city scenes, reconstructs the hit geometry (planar face or diffracting edge)
in the `(object_origins, object_vectors)` convention used by `fpt_jax.trace_rays`,
and cross-validates every candidate path against physical interaction laws
(equal angles for reflection, diffraction Keller cone, and transmission).
Only validated paths are written to the output TOML file, deduplicated to
one example per distinct interaction-type sequence (up to 3 bounces) per
scene/config.
"""

import logging
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import mitsuba as mi
import numpy as np
from sionna import rt

from .checks import check_path_interactions

logger = logging.getLogger(__name__)

OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "sionna_paths.toml"

# Sionna's `InteractionType` enum values (see `sionna.rt.InteractionType`).
SPECULAR = 1
DIFFRACTION = 8
REFRACTION = 4

TYPE_NAMES = {
    SPECULAR: "reflection",
    DIFFRACTION: "diffraction",
    REFRACTION: "transmission",
}
TYPE_TO_ENUM = {"reflection": 0, "diffraction": 1, "transmission": 2}

# (scene, tag, tx, rx, max_depth) configurations to sample paths from.
HEADER = """\
# Benchmark path-tracing test cases extracted from Sionna RT (https://nvlabs.github.io/sionna/rt/)
# built-in city scenes, covering realistic multi-bounce reflection, diffraction,
# and transmission paths. Regenerate with `pytest --generate-sionna-dataset`.
#
# For each Sionna scene, `sionna.rt.PathSolver` was run with reflection,
# diffraction (incl. free-edge diffraction), and refraction/transmission all
# enabled. For every valid, non-line-of-sight path found, the hit geometry
# (mesh face, for reflection/transmission; mesh edge, for diffraction) was
# reconstructed as an (origin, vectors) pair in the same convention as
# `tests/data/simple_paths.toml`, and cross-validated against the physical laws
# governing each interaction type (law of reflection, Keller's law of diffraction,
# and transmission/refraction across interfaces). This makes each case an
# independent check based on physics laws on realistic building/street geometry,
# not just on synthetic examples.
#
# One example test case per distinct interaction-type sequence (up to 3 bounces)
# was kept per scene; see `description` for the originating scene/scenario and
# the objects involved. See `tests/data/simple_paths.toml` for the field schema.
#
# Note: transmission through a slab (e.g. window glass) naturally produces two
# consecutive "transmission" interactions (entry and exit face of the slab).
"""

SCENE_CONFIGS = [
    ("simple_street_canyon", "los_reflection", [-20.0, 0.0, 15.0], [20.0, 5.0, 1.5], 3),
    (
        "simple_street_canyon",
        "corner_diffraction",
        [-40.0, -20.0, 8.0],
        [40.0, 25.0, 1.5],
        3,
    ),
    (
        "simple_street_canyon",
        "window_transmission",
        [-10.0, -3.0, 2.0],
        [10.0, -3.0, 2.0],
        3,
    ),
    ("simple_street_canyon", "mixed_1", [-60.0, -30.0, 20.0], [60.0, 30.0, 1.5], 3),
    ("simple_street_canyon", "mixed_2", [-30.0, 10.0, 10.0], [30.0, -15.0, 2.0], 3),
    ("etoile", "plaza_1", [0.0, 0.0, 30.0], [80.0, 40.0, 1.5], 2),
    ("etoile", "plaza_2", [0.0, 0.0, 25.0], [-60.0, 60.0, 1.5], 2),
    ("etoile", "plaza_3", [30.0, -30.0, 20.0], [-40.0, 50.0, 1.5], 2),
]


def object_by_id(scene: rt.Scene) -> dict[int, rt.SceneObject]:
    return {obj.object_id: obj for obj in scene.objects.values()}


def mesh_centroid(obj: rt.SceneObject) -> tuple[float, float, float]:
    mesh = obj.mi_mesh
    pts = np.stack(
        [
            np.array(mesh.vertex_position(i)).reshape(-1)
            for i in range(mesh.vertex_count())
        ]
    )
    c = pts.mean(axis=0)
    return (round(float(c[0]), 6), round(float(c[1]), 6), round(float(c[2]), 6))


def object_display_names(scene: rt.Scene) -> dict[int, str]:
    """Map `object_id` to a display name for the `description` field.

    Sionna assigns "no-name-N" identifiers to anonymous mesh shapes in the
    order it happens to iterate `scene.shapes()`, and `object_id` is a
    reinterpreted Mitsuba shape pointer -- neither is stable across process
    runs (Mitsuba's scene loader does not guarantee shape iteration order),
    so relying on either here would make this purely cosmetic label churn on
    every regeneration. Renumber anonymous objects by mesh centroid instead,
    a property of the geometry itself, so the label is reproducible.
    """
    named: dict[int, str] = {}
    anonymous: list[rt.SceneObject] = []
    for name, obj in scene.objects.items():
        if name.startswith("no-name-"):
            anonymous.append(obj)
        else:
            named[obj.object_id] = name
    anonymous.sort(key=mesh_centroid)
    for i, obj in enumerate(anonymous):
        named[obj.object_id] = f"unnamed-{i}"
    return named


def face_vertices(obj: rt.SceneObject, prim: int) -> np.ndarray:
    mesh = obj.mi_mesh
    idx = np.array(mesh.face_indices(int(prim))).reshape(-1)
    return np.stack([np.array(mesh.vertex_position(int(i))).reshape(-1) for i in idx])


def reflection_or_transmission_plane(
    obj: rt.SceneObject, prim: int
) -> tuple[np.ndarray, np.ndarray]:
    """A planar face is spanned by two of its edges, from one of its vertices."""
    v = face_vertices(obj, prim)
    return v[0], np.stack([v[1] - v[0], v[2] - v[0]])


def best_edge_for_point(
    obj: rt.SceneObject, prim: int, point: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray], float]:
    """Find which of the hit face's 3 edges the diffraction point lies on.

    Sionna does not directly expose the wedge (edge) index of a diffraction
    interaction, only the reference face. We recover it by checking, for each
    of that face's 3 edges, the perpendicular distance from the recorded
    interaction point to the edge's line -- the correct edge has ~zero residual.
    """
    v = face_vertices(obj, prim)
    best, best_res = None, None
    for i in range(3):
        origin, direction = v[i], v[(i + 1) % 3] - v[i]
        d = direction / np.linalg.norm(direction)
        perp = (point - origin) - np.dot(point - origin, d) * d
        res = float(np.linalg.norm(perp))
        if best_res is None or res < best_res:
            best, best_res = (origin, direction), res
    assert best is not None and best_res is not None
    return best, best_res


def load_scene(scene_name: str) -> rt.Scene:
    scene = rt.load_scene(getattr(rt.scene, scene_name))
    scene.tx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V"
    )
    scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V"
    )
    return scene


def compute_paths(
    scene: rt.Scene, tx_pos: list[float], rx_pos: list[float], max_depth: int
) -> rt.Paths:
    scene.add(rt.Transmitter("tx", position=mi.Point3f(*tx_pos)))
    scene.add(rt.Receiver("rx", position=mi.Point3f(*rx_pos)))
    paths = rt.PathSolver()(
        scene,
        max_depth=max_depth,
        los=False,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=True,
        diffraction=True,
        edge_diffraction=True,
        seed=42,
    )
    scene.remove("tx")
    scene.remove("rx")
    return paths


def extract_records(
    scene_name: str,
    scene: rt.Scene,
    paths: rt.Paths,
    tx_pos: list[float],
    rx_pos: list[float],
    tag: str,
) -> list[dict[str, Any]]:
    """Turn every valid, non-LOS path into a record with reconstructed geometry."""
    objs = object_by_id(scene)
    names = object_display_names(scene)
    vertices = paths.vertices.numpy()  # (depth, rx, tx, path, 3)
    interactions = paths.interactions.numpy()
    objects = paths.objects.numpy()
    primitives = paths.primitives.numpy()
    valid = paths.valid.numpy()

    records = []
    for p in range(vertices.shape[3]):
        if not valid[0, 0, p]:
            continue
        depth_types = interactions[:, 0, 0, p]
        n_int = int(np.sum(depth_types != 0))
        if n_int == 0:
            continue  # pure line-of-sight, not interesting for this dataset

        record: dict[str, Any] = {
            "scene": scene_name,
            "tag": tag,
            "tx": tx_pos,
            "rx": rx_pos,
            "interactions": [],
        }
        ok = True
        for d in range(n_int):
            itype = int(depth_types[d])
            oid = int(objects[d, 0, 0, p])
            prim = int(primitives[d, 0, 0, p])
            vtx = vertices[d, 0, 0, p].astype(float)
            if oid not in objs or itype not in TYPE_NAMES:
                ok = False
                break
            obj = objs[oid]
            if itype in (SPECULAR, REFRACTION):
                origin, vecs = reflection_or_transmission_plane(obj, prim)
            else:  # DIFFRACTION
                (origin, direction), residual = best_edge_for_point(obj, prim, vtx)
                vecs = direction[None, :]
                if residual > 1e-3:
                    ok = False
                    break
            record["interactions"].append(
                {
                    "type": TYPE_NAMES[itype],
                    "object_name": names.get(oid),
                    "vertex": vtx.tolist(),
                    "origin": origin.tolist(),
                    "vectors": vecs.tolist(),
                }
            )
        if ok and len(record["interactions"]) == n_int:
            records.append(record)
    return records


def build_arrays(record: dict[str, Any]) -> tuple[list, list, list]:
    interactions = record["interactions"]
    max_dims = max(len(ix["vectors"]) for ix in interactions)
    object_origins, object_vectors, interaction_list = [], [], []
    for ix in interactions:
        object_origins.append(ix["origin"])
        vecs = list(ix["vectors"])
        while len(vecs) < max_dims:
            vecs.append([0.0, 0.0, 0.0])
        object_vectors.append(vecs)
        interaction_list.append(TYPE_TO_ENUM[ix["type"]])
    return object_origins, object_vectors, interaction_list


def candidate_sort_key(record: dict[str, Any]) -> tuple[float, ...]:
    """A deterministic tie-break for candidates within a (scene, signature)
    group, derived from the candidate's own geometry.

    `sionna.rt.PathSolver` runs on a JIT-compiled, parallel (Dr.Jit) backend,
    so the order in which it returns candidate paths is not guaranteed to be
    stable across runs even for a fixed `seed`. Without this, "keep the first
    validated candidate" would pick a different (still valid) candidate on
    each run.
    """
    return tuple(v for ix in record["interactions"] for v in ix["vertex"])


def validate(record: dict[str, Any], atol: float = 1e-2) -> bool:
    """Check that the path satisfies physical interaction laws (reflection, diffraction, transmission)."""
    object_origins, object_vectors, interaction_list = build_arrays(record)
    expected = [ix["vertex"] for ix in record["interactions"]]
    return check_path_interactions(
        record["tx"],
        record["rx"],
        expected,
        object_origins,
        object_vectors,
        interaction_list,
        atol=atol,
    )


def fmt_vec(v: list[float]) -> str:
    return "[" + ", ".join(f"{x:.10g}" for x in v) + "]"


def fmt_mat(m: list[list[float]]) -> str:
    return "[" + ", ".join(fmt_vec(row) for row in m) + "]"


def to_toml_case(name: str, record: dict[str, Any], sig: tuple[str, ...]) -> str:
    object_origins, object_vectors, interaction_list = build_arrays(record)
    expected_path = [ix["vertex"] for ix in record["interactions"]]
    objects_str = ", ".join(ix["object_name"] or "?" for ix in record["interactions"])
    description = (
        f"Sionna RT scene '{record['scene']}' ({record['tag']}): "
        + " -> ".join(sig)
        + f" path (objects: {objects_str})."
    )
    lines = [
        "[[testcase]]",
        f'name = "{name}"',
        f'description = "{description}"',
        f'source_scene = "{record["scene"]}"',
        f"tx = {fmt_vec(record['tx'])}",
        f"rx = {fmt_vec(record['rx'])}",
        f"object_origins = {fmt_mat(object_origins)}",
        "object_vectors = [" + ", ".join(fmt_mat(v) for v in object_vectors) + "]",
        f"interaction_list = {interaction_list}",
        f"expected_path = {fmt_mat(expected_path)}",
        "",
    ]
    return "\n".join(lines)


def generate_sionna_dataset() -> None:
    """Regenerate `tests/data/sionna_paths.toml` from Sionna RT city scenes."""
    all_records: list[dict[str, Any]] = []
    for scene_name, tag, tx_pos, rx_pos, max_depth in SCENE_CONFIGS:
        scene = load_scene(scene_name)
        paths = compute_paths(scene, tx_pos, rx_pos, max_depth)
        records = extract_records(scene_name, scene, paths, tx_pos, rx_pos, tag)
        logger.info("%s/%s: %d candidate paths", scene_name, tag, len(records))
        all_records.extend(records)

    # Group by (scene, interaction-type signature); keep one validated example each.
    groups: dict[tuple[str, tuple[str, ...]], list[int]] = defaultdict(list)
    for i, r in enumerate(all_records):
        sig = tuple(ix["type"] for ix in r["interactions"])
        groups[(r["scene"], sig)].append(i)

    selected: OrderedDict[tuple[str, tuple[str, ...]], dict[str, Any]] = OrderedDict()
    for scene_name, sig in sorted(groups):
        idxs = sorted(
            groups[(scene_name, sig)], key=lambda i: candidate_sort_key(all_records[i])
        )
        for i in idxs:
            record = all_records[i]
            if validate(record):
                selected[(scene_name, sig)] = record
                break
        else:
            logger.warning("no candidate validated for %s %s", scene_name, sig)

    logger.info(
        "Validated %d / %d distinct (scene, signature) combinations.",
        len(selected),
        len(groups),
    )

    counts: dict[str, int] = {}
    out = []
    for (scene_name, sig), record in selected.items():
        sig_short = "_".join(t[:4] for t in sig)
        name = f"{scene_name}__{record['tag']}__{sig_short}"
        counts[name] = counts.get(name, 0) + 1
        if counts[name] > 1:
            name = f"{name}_{counts[name]}"
        out.append(to_toml_case(name, record, sig))

    OUTPUT_PATH.write_text(HEADER + "\n" + "\n".join(out), encoding="utf-8")
    logger.info("Wrote %d test cases to %s", len(out), OUTPUT_PATH)
