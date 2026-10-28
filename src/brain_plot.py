"""Grafo / cerebro 3D estilo oscuro (Plotly) para la app."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def brain_graph_dark(
    title: str = "Grosor cortical medio (muestra piloto HCP)",
    n_nodes: int = 68,
    seed: int = 2026,
) -> go.Figure:
    """
    Visualización 3D didáctica de nodos regionales sobre una nube cerebral.
    No requiere los 30 GB de HCP: usa geometría sintética orientativa
    (el grosor real entra en el modelo tabular, no en cada vértice aquí).

    Para un grafo con superficies HCP reales del sujeto 100206, usar el
    notebook Colab de referencia del usuario (CIFTI + midthickness).
    """
    rng = np.random.default_rng(seed)

    # Nube de puntos con forma aproximada de dos hemisferios
    n_half = n_nodes // 2
    left = rng.normal(size=(n_half, 3)) * np.array([25, 35, 30]) + np.array([-35, 0, 0])
    right = rng.normal(size=(n_nodes - n_half, 3)) * np.array([25, 35, 30]) + np.array([35, 0, 0])
    coords = np.vstack([left, right])

    # Color = "grosor relativo" ilustrativo
    color = rng.normal(2.65, 0.12, size=n_nodes)
    color = np.clip(color, 2.3, 3.0)

    # Aristas k-NN ligeras
    from scipy.spatial.distance import cdist

    D = cdist(coords, coords)
    np.fill_diagonal(D, np.inf)
    k = 4
    edge_x, edge_y, edge_z = [], [], []
    for i in range(n_nodes):
        js = np.argsort(D[i])[:k]
        for j in js:
            if i < j:
                edge_x += [coords[i, 0], coords[j, 0], None]
                edge_y += [coords[i, 1], coords[j, 1], None]
                edge_z += [coords[i, 2], coords[j, 2], None]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(color="rgba(100,140,200,0.35)", width=2),
            name="Vecindad regional",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker=dict(
                size=5,
                color=color,
                colorscale="Viridis",
                colorbar=dict(title="Grosor (ilustrativo)"),
                opacity=0.95,
            ),
            name="ROIs",
            text=[f"ROI {i+1}<br>grosor≈{c:.2f}" for i, c in enumerate(color)],
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        width=900,
        height=650,
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=1.55, y=1.45, z=1.1)),
        ),
        legend=dict(x=0.01, y=0.99),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
    )
    return fig


def try_nilearn_surface_figure(title: str = "Superficie cortical (nilearn)") -> go.Figure | None:
    """
    Intenta cargar fsaverage5 con nilearn para un mesh más realista.
    Si falla (sin red en el build), devuelve None.
    """
    try:
        from nilearn import datasets, surface

        fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
        fig = go.Figure()
        for hemi, key in [("left", "pial_left"), ("right", "pial_right")]:
            mesh = surface.load_surf_mesh(fsavg[key])
            xyz = np.asarray(mesh.coordinates, dtype=float)
            faces = np.asarray(mesh.faces, dtype=int)
            faces = faces[::4]  # aligerar
            fig.add_trace(
                go.Mesh3d(
                    x=xyz[:, 0],
                    y=xyz[:, 1],
                    z=xyz[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color="#AEBAC7",
                    opacity=0.22,
                    name=f"Superficie {hemi}",
                    hoverinfo="skip",
                    showlegend=True,
                )
            )
        fig.update_layout(
            title=title,
            template="plotly_dark",
            width=900,
            height=650,
            margin=dict(l=0, r=0, t=50, b=0),
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
                camera=dict(eye=dict(x=1.6, y=1.5, z=1.15)),
            ),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
        )
        return fig
    except Exception:
        return None
