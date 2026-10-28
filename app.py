"""
App Streamlit — Delay Discounting y grosor cortical (HCP piloto)
Modelo A (real, verde) | Modelo B (demo FC sintética, rojo)
Grafo 3D estilo Colab (superficie + aristas rojo/azul)
"""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.model import fit_model_a, fit_model_b, model_table, fit_stats
from src.brain_plot import brain_graph_dark, load_precomputed_graph, build_nilearn_style_graph

st.set_page_config(
    page_title="HCP Delay Discounting · Grosor cortical",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = ROOT / "data" / "subjects_features_real.csv"
DATA_DIR = ROOT / "data"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Subject", "DDisc_AUC_200", "Thickness_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")
    return df


@st.cache_resource
def cached_brain_figure(mode: str):
    if mode == "precomputed":
        fig = load_precomputed_graph(DATA_DIR)
        if fig is not None:
            return fig, "precomputed"
        mode = "nilearn"
    if mode == "nilearn":
        try:
            return build_nilearn_style_graph(), "nilearn"
        except Exception:
            return brain_graph_dark(), "fallback"
    return brain_graph_dark(), "fallback"


def style_header_green(text: str):
    st.markdown(
        f"<h2 style='color:#2ecc71;margin-bottom:0.2rem;'>{text}</h2>",
        unsafe_allow_html=True,
    )


def style_header_red(text: str):
    st.markdown(
        f"<h2 style='color:#e74c3c;margin-bottom:0.2rem;'>{text}</h2>",
        unsafe_allow_html=True,
    )


st.sidebar.title("Navegación")
page = st.sidebar.radio(
    "Sección",
    [
        "Inicio",
        "Modelo A — Real (verde)",
        "Modelo B — Demo FC (rojo)",
        "Grafo 3D",
        "Datos y limitaciones",
    ],
)
st.sidebar.markdown("---")
st.sidebar.info(
    "Muestra piloto HCP-YA (~21 sujetos). "
    "Solo Structural + conducta; sin resting-state completo de la cohorte."
)

if not DATA_PATH.exists():
    st.error(
        f"No se encuentra `{DATA_PATH}`.\n\n"
        "Copia tu `subjects_features_real.csv` a la carpeta `data/` del repo."
    )
    st.stop()

df = load_data(str(DATA_PATH))

if page == "Inicio":
    st.title("Neuroeconometría · Delay Discounting y grosor cortical")
    st.markdown(
        """
        Pipeline con **datos reales** del Human Connectome Project (piloto).

        | Modelo | Contenido |
        |--------|-----------|
        | **A (verde)** | `AUC_200 ~ grosor + sexo + edad` — 100 % HCP |
        | **B (rojo)** | Igual + **FC sintética** — solo demostración metodológica |

        ### Por qué dos modelos
        - El resting-state de ~21 sujetos (~227 GB) no cabía en disco.
        - Se priorizó el paquete **Structural** (~30 GB) para grosor real.
        - El Modelo B ilustra el pipeline con FC, dejando claro que no es observada.
        """
    )
    st.subheader("Vista previa de los datos")
    st.dataframe(df, use_container_width=True)
    st.caption(f"N = {len(df)} sujetos · archivo: `data/subjects_features_real.csv`")

elif page == "Modelo A — Real (verde)":
    style_header_green("Modelo A — Datos 100 % reales (HCP)")
    st.markdown(
        """
        **Especificación**

        $$
        \\mathrm{DDisc\\_AUC\\_200}_i =
        \\beta_0 + \\beta_1\\,\\mathrm{Thickness\\_mean}_i
        + \\beta_2\\,\\mathrm{Gender\\_M}_i
        + \\beta_3\\,\\mathrm{Age\\_mid}_i + \\varepsilon_i
        $$

        Unidad de análisis: **participante**.
        """
    )
    model, data = fit_model_a(df)
    stats = fit_stats(model)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("N", stats["N"])
    c2.metric("R²", f"{stats['R²']:.3f}")
    c3.metric("R² ajustado", f"{stats['R² ajustado']:.3f}")
    c4.metric("AIC", f"{stats['AIC']:.1f}")
    st.subheader("Coeficientes")
    st.dataframe(
        model_table(model).style.format(
            {"Coeficiente": "{:.4f}", "Error std.": "{:.4f}", "t": "{:.3f}", "p-valor": "{:.4f}"}
        ),
        use_container_width=True,
    )
    st.subheader("Dispersión: AUC vs grosor")
    fig = px.scatter(
        data, x="Thickness_mean", y="DDisc_AUC_200", color="Gender",
        hover_data=["Subject", "Age"], trendline="ols", template="plotly_dark",
        labels={"Thickness_mean": "Grosor cortical medio (mm)", "DDisc_AUC_200": "AUC ($200)"},
        title="Asociación observada (muestra piloto)",
    )
    fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Interpretación breve"):
        st.markdown(
            """
            - N ≈ 20 es **exploratorio**: no generalizar a toda la cohorte HCP.
            - La edad es el punto medio del rango público, no la edad exacta.
            """
        )

elif page == "Modelo B — Demo FC (rojo)":
    style_header_red("Modelo B — Demostración con FC sintética")
    st.warning(
        "**La conectividad funcional de este modelo NO proviene de HCP.** "
        "Es sintética: solo ilustra el pipeline (grosor + FC)."
    )
    model, data = fit_model_b(df)
    stats = fit_stats(model)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("N", stats["N"])
    c2.metric("R²", f"{stats['R²']:.3f}")
    c3.metric("R² ajustado", f"{stats['R² ajustado']:.3f}")
    c4.metric("AIC", f"{stats['AIC']:.1f}")
    st.subheader("Coeficientes (no interpretar FC como resultado empírico)")
    st.dataframe(
        model_table(model).style.format(
            {"Coeficiente": "{:.4f}", "Error std.": "{:.4f}", "t": "{:.3f}", "p-valor": "{:.4f}"}
        ),
        use_container_width=True,
    )
    fig = px.scatter(
        data, x="FC_synthetic", y="DDisc_AUC_200", color="Thickness_mean",
        hover_data=["Subject"], template="plotly_dark",
        labels={"FC_synthetic": "FC sintética (z)", "DDisc_AUC_200": "AUC $200"},
        title="Demo — no usar para conclusiones sobre FC real",
    )
    fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Grafo 3D":
    st.title("Visualización 3D")
    with st.spinner("Generando grafo 3D..."):
        fig, _used = cached_brain_figure("precomputed")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.title("Datos, entrega y limitaciones")
    st.markdown(
        f"""
        ### Archivo de datos
        - `data/subjects_features_real.csv` · **{len(df)}** filas
        - Columnas: `{', '.join(df.columns)}`

        ### Limitaciones
        - N pequeño → resultados exploratorios
        - Modelo B: FC sintética (título rojo)
        - Grafo 3D: estilo visual de referencia; aristas de FC ilustrativa si no hay export precomputado
        """
    )
    st.dataframe(df.describe(include="all"), use_container_width=True)
