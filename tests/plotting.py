"""Optional interactive 3D visualization of benchmark dataset test cases, for
debugging failures with `pytest --plot-failures` (see `tests/conftest.py`).

Uses Plotly directly, which renders large 3D meshes far better than
Matplotlib's 3D backend. Both `plotly` and, for cases sourced from a Sionna RT
scene, `sionna-rt` are imported lazily, so neither is required to run the test
suite normally.
"""

from pathlib import Path
from typing import Any
import plotly.graph_objects as go
from sionna import rt

import numpy as np


def _scene_mesh(scene_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Combined (vertices, triangles) arrays for a Sionna RT scene's meshes."""

    scene = rt.load_scene(getattr(rt.scene, scene_name))
    all_vertices = []
    all_triangles = []
    offset = 0
    for obj in scene.objects.values():
        mesh = obj.mi_mesh
        num_vertices = mesh.vertex_count()
        vertices = np.array(
            [np.array(mesh.vertex_position(i)).reshape(-1) for i in range(num_vertices)]
        )
        triangles = np.array(
            [
                np.array(mesh.face_indices(prim)).reshape(-1)
                for prim in range(mesh.face_count())
            ]
        )
        all_vertices.append(vertices)
        all_triangles.append(triangles + offset)
        offset += num_vertices

    return np.concatenate(all_vertices, axis=0), np.concatenate(all_triangles, axis=0)


def _path_points(
    tx: np.ndarray, rx: np.ndarray, interactions: np.ndarray
) -> np.ndarray:
    return np.concatenate([tx[None, :], interactions, rx[None, :]], axis=0)


def _dummy_object_traces(case: dict[str, Any], expected: np.ndarray) -> list[Any]:
    """Draw a unit-length edge (diffraction) or unit-sized plane patch
    (reflection/transmission) for each interaction, centered on its expected
    interaction point -- since no real scene geometry is available for
    hand-crafted (non-Sionna) test cases, but `object_vectors` alone doesn't
    say where along the corresponding infinite line/plane the interaction
    actually happens, or how "large" it is.
    """

    object_vectors = np.asarray(case["object_vectors"], dtype=float)

    traces = []
    for i, vectors in enumerate(object_vectors):
        center = expected[i]
        # Basis vectors are zero-padded when this case mixes edges
        # (num_dims=1) with planes (num_dims=2); drop the padding.
        directions = [
            v / np.linalg.norm(v) for v in vectors if np.linalg.norm(v) > 1e-9
        ]
        show_legend = i == 0

        if len(directions) == 1:
            (d,) = directions
            p0, p1 = center - 0.5 * d, center + 0.5 * d
            traces.append(
                go.Scatter3d(
                    x=[p0[0], p1[0]],
                    y=[p0[1], p1[1]],
                    z=[p0[2], p1[2]],
                    mode="lines",
                    line={"color": "royalblue", "width": 8},
                    name="edge (dummy)",
                    legendgroup="objects",
                    showlegend=show_legend,
                    hoverinfo="skip",
                )
            )
        elif len(directions) >= 2:
            v1, v2 = directions[0], directions[1]
            corners = [
                center - 0.5 * v1 - 0.5 * v2,
                center + 0.5 * v1 - 0.5 * v2,
                center + 0.5 * v1 + 0.5 * v2,
                center - 0.5 * v1 + 0.5 * v2,
            ]
            traces.append(
                go.Mesh3d(
                    x=[c[0] for c in corners],
                    y=[c[1] for c in corners],
                    z=[c[2] for c in corners],
                    i=[0, 0],
                    j=[1, 2],
                    k=[2, 3],
                    color="royalblue",
                    opacity=0.3,
                    name="plane (dummy)",
                    legendgroup="objects",
                    showlegend=show_legend,
                    hoverinfo="skip",
                    flatshading=True,
                )
            )
    return traces


def plot_testcase(case: dict[str, Any], got_path: np.ndarray, out_dir: Path) -> Path:
    """Save an interactive 3D plot comparing `case["expected_path"]` against `got_path`.

    If `case` has a `source_scene` field (see `tests/data/sionna_paths.toml`)
    and `sionna-rt` is installed, the scene's mesh is also drawn.
    """
    import plotly.graph_objects as go

    tx = np.asarray(case["tx"], dtype=float)
    rx = np.asarray(case["rx"], dtype=float)
    expected = np.asarray(case["expected_path"], dtype=float)
    got = np.asarray(got_path, dtype=float)

    fig = go.Figure()

    scene_name = case.get("source_scene")
    if scene_name is None:
        for trace in _dummy_object_traces(case, expected):
            fig.add_trace(trace)
    else:
        mesh = _scene_mesh(scene_name)

        vertices, triangles = mesh
        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                color="lightgray",
                opacity=0.4,
                name=scene_name,
                hoverinfo="skip",
                showlegend=True,
                flatshading=True,
            )
        )

    def add_path(points: np.ndarray, name: str, color: str, dash: str | None) -> None:
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="lines+markers",
                marker={"size": 3, "color": color},
                line={"color": color, "width": 5, "dash": dash},
                name=name,
            )
        )

    add_path(_path_points(tx, rx, expected), "expected_path", "green", None)
    add_path(_path_points(tx, rx, got), "trace_rays", "red", "dash")

    def add_marker(point: np.ndarray, name: str, symbol: str) -> None:
        fig.add_trace(
            go.Scatter3d(
                x=[point[0]],
                y=[point[1]],
                z=[point[2]],
                mode="markers+text",
                marker={"size": 6, "color": "black", "symbol": symbol},
                text=[name],
                name=name,
            )
        )

    add_marker(tx, "tx", "diamond")
    add_marker(rx, "rx", "square")

    fig.update_layout(title=case["name"], scene_aspectmode="data")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case['name']}.html"
    fig.write_html(out_path)
    return out_path
