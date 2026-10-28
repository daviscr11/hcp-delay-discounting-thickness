"""
Grafo cerebral 3D estilo oscuro (Plotly), alineado con el notebook Colab:
superficie cortical + ROIs + aristas (positivas en rojo, negativas en azul).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import plotly.graph_objects as go


def _edges_from_matrix(C: np.ndarray, coords: np.ndarray, n_edges: int = 160):
    n = C.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    vals = C[iu, ju]
    order = np.argsort(np.abs(vals))[-min(n_edges, len(vals)):]
    pos_x, pos_y, pos_z = [], [], []
    neg_x, neg_y, neg_z = [], [], []
    for q in order:
        i, j = int(iu[q]), int(ju[q])
        v = float(vals[q])
        xs = [coords[i, 0], coords[j, 0], None]
        ys = [coords[i, 1], coords[j, 1], None]
        zs = [coords[i, 2], coords[j, 2], None]
        if v >= 0:
            pos_x += xs
            pos_y += ys
            pos_z += zs
        else:
            neg_x += xs
            neg_y += ys
            neg_z += zs
    return (pos_x, pos_y, pos_z), (neg_x, neg_y, neg_z)


def figure_from_arrays(
    coords,
    strength,
    C=None,
    labels=None,
    mesh_left=None,
    mesh_right=None,
    title="Conectividad funcional — grafo 3D",
    n_edges=160,
):
    coords = np.asarray(coords, dtype=float)
    strength = np.asarray(strength, dtype=float).ravel()
    n = coords.shape[0]
    if labels is None:
        labels = [f"ROI {i}" for i in range(n)]

    fig = go.Figure()

    for mesh, name in [(mesh_left, "Hemisferio izquierdo"), (mesh_right, "Hemisferio derecho")]:
        if mesh is None:
            continue
        xyz, faces = mesh
        xyz = np.asarray(xyz, dtype=float)
        faces = np.asarray(faces, dtype=int)
        if len(faces) > 8000:
            faces = faces[:: max(1, len(faces) // 6000)]
        fig.add_trace(
            go.Mesh3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color="#AEBAC7", opacity=0.16, name=name,
                hoverinfo="skip", showlegend=True,
                lighting=dict(ambient=0.72, diffuse=0.45, specular=0.1),
            )
        )

    if C is not None:
        (px, py, pz), (nx, ny, nz) = _edges_from_matrix(C, coords, n_edges=n_edges)
        if px:
            fig.add_trace(
                go.Scatter3d(
                    x=px, y=py, z=pz, mode="lines",
                    line=dict(color="rgba(220,70,50,0.55)", width=2),
                    name="Correlaciones positivas", hoverinfo="skip",
                )
            )
        if nx:
            fig.add_trace(
                go.Scatter3d(
                    x=nx, y=ny, z=nz, mode="lines",
                    line=dict(color="rgba(70,120,220,0.50)", width=2),
                    name="Correlaciones negativas", hoverinfo="skip",
                )
            )

    cmin, cmax = float(np.nanmin(strength)), float(np.nanmax(strength))
    if not np.isfinite(cmin) or not np.isfinite(cmax) or cmin == cmax:
        cmin, cmax = 0.0, 1.0

    fig.add_trace(
        go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers",
            marker=dict(
                size=4.5, color=strength, colorscale="Viridis",
                cmin=cmin, cmax=cmax,
                colorbar=dict(title="Fuerza FC media", len=0.65),
                opacity=0.95,
            ),
            text=[f"{lab}<br>FC media abs={s:.3f}" for lab, s in zip(labels, strength)],
            hovertemplate="%{text}<extra></extra>",
            name="ROIs: fuerza FC",
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        width=1100, height=750,
        margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.5, z=1.15)),
        ),
        legend=dict(x=0.01, y=0.99),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
    )
    return fig


def load_precomputed_graph(data_dir: Path):
    data_dir = Path(data_dir)
    coords_p = data_dir / "graph_coords.npy"
    strength_p = data_dir / "graph_strength.npy"
    fc_p = data_dir / "graph_fc.npy"
    if not (coords_p.exists() and strength_p.exists() and fc_p.exists()):
        return None
    coords = np.load(coords_p)
    strength = np.load(strength_p)
    C = np.load(fc_p)
    labels = None
    lab_p = data_dir / "graph_labels.txt"
    if lab_p.exists():
        labels = [ln.strip() for ln in lab_p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return figure_from_arrays(
        coords, strength, C=C, labels=labels,
        title="Conectividad funcional real en reposo (export Colab)",
        n_edges=180,
    )


def build_nilearn_style_graph(seed: int = 2026, n_edges: int = 160):
    from nilearn import datasets, surface
    from scipy.spatial.distance import cdist

    rng = np.random.default_rng(seed)
    fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    atlas = datasets.fetch_atlas_surf_destrieux()

    meshes = {}
    nodes = []
    for hemi, mesh_key, map_key in [
        ("left", "pial_left", "map_left"),
        ("right", "pial_right", "map_right"),
    ]:
        mesh = surface.load_surf_mesh(fsavg[mesh_key])
        xyz = np.asarray(mesh.coordinates, dtype=float)
        faces = np.asarray(mesh.faces, dtype=int)
        meshes[hemi] = (xyz, faces)
        labels_v = np.asarray(atlas[map_key]).astype(int).ravel()
        for code in sorted(np.unique(labels_v)):
            if code == 0:
                continue
            idx = np.flatnonzero(labels_v == code)
            if len(idx) < 8:
                continue
            centroid = xyz[idx].mean(axis=0)
            nodes.append({"hemi": hemi, "code": int(code), "xyz": centroid, "label": f"{hemi[0].upper()}_{code}"})

    coords = np.vstack([n["xyz"] for n in nodes])
    labels = [n["label"] for n in nodes]
    n = len(nodes)

    n_lat = 4
    loads = rng.normal(size=(n, n_lat))
    D = cdist(coords, coords)
    K = np.exp(-D / 28.0)
    np.fill_diagonal(K, 0.0)
    loads = K @ loads
    loads = (loads - loads.mean(0)) / (loads.std(0) + 1e-8)
    T = 120
    latent = rng.normal(size=(n_lat, T))
    ts = loads @ latent + rng.normal(scale=0.35, size=(n, T))
    C = np.corrcoef(ts)
    np.fill_diagonal(C, 0.0)
    strength = np.mean(np.abs(C), axis=1)

    return figure_from_arrays(
        coords, strength, C=C, labels=labels,
        mesh_left=meshes["left"], mesh_right=meshes["right"],
        title="Grafo 3D estilo HCP · superficie real + FC ilustrativa (no resting-state HCP)",
        n_edges=n_edges,
    )


def brain_graph_dark(title: str = "Grafo cerebral 3D", **kwargs):
    try:
        return build_nilearn_style_graph()
    except Exception:
        rng = np.random.default_rng(2026)
        n_nodes = 68
        left = rng.normal(size=(n_nodes // 2, 3)) * np.array([25, 35, 30]) + np.array([-35, 0, 0])
        right = rng.normal(size=(n_nodes - n_nodes // 2, 3)) * np.array([25, 35, 30]) + np.array([35, 0, 0])
        coords = np.vstack([left, right])
        from scipy.spatial.distance import cdist
        D = cdist(coords, coords)
        np.fill_diagonal(D, np.inf)
        C = np.exp(-D / 40.0)
        np.fill_diagonal(C, 0.0)
        signs = rng.choice([-1.0, 1.0], size=C.shape)
        C = C * signs
        C = (C + C.T) / 2.0
        strength = np.mean(np.abs(C), axis=1)
        return figure_from_arrays(coords, strength, C=C, title=title, n_edges=120)
