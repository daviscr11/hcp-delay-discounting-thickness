"""
App Streamlit — Delay Discounting y grosor cortical (HCP piloto)
Modelo A (real, verde) | Modelo B (demo FC sintética, rojo)
"""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.model import fit_model_a, fit_model_b, model_table, fit_stats
from src.brain_plot import brain_graph_dark, try_nilearn_surface_figure

st.set_page_config(
    page_title="HCP Delay Discounting · Grosor cortical",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = ROOT / "data" / "subjects_features_real.csv"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Subject", "DDisc_AUC_200", "Thickness_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")
    return df


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


# ----- Sidebar -----
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
    "Solo Structural + conducta; sin resting-state completo."
)

# ----- Data -----
if not DATA_PATH.exists():
    st.error(
        f"No se encuentra `{DATA_PATH}`.\n\n"
        "Copia tu `subjects_features_real.csv` a la carpeta `data/` del repo."
    )
    st.stop()

df = load_data(str(DATA_PATH))

# ===================== INICIO =====================
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
        - El Modelo B muestra cómo quedaría el análisis con conectividad,
          dejando **explícito** que la FC no es observada.
        """
    )
    st.subheader("Vista previa de los datos")
    st.dataframe(df, use_container_width=True)
    st.caption(f"N = {len(df)} sujetos · archivo: `data/subjects_features_real.csv`")

# ===================== MODELO A =====================
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
        Sin datos sintéticos en predictores ni en la variable dependiente.
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
    st.dataframe(model_table(model).style.format({
        "Coeficiente": "{:.4f}",
        "Error std.": "{:.4f}",
        "t": "{:.3f}",
        "p-valor": "{:.4f}",
    }), use_container_width=True)

    st.subheader("Dispersión: AUC vs grosor")
    fig = px.scatter(
        data,
        x="Thickness_mean",
        y="DDisc_AUC_200",
        color="Gender",
        hover_data=["Subject", "Age"],
        trendline="ols",
        template="plotly_dark",
        labels={
            "Thickness_mean": "Grosor cortical medio (mm)",
            "DDisc_AUC_200": "Delay Discounting AUC ($200)",
        },
        title="Asociación observada (muestra piloto)",
    )
    fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Interpretación breve"):
        st.markdown(
            """
            - Un **p-valor** alto en grosor indica que, en esta muestra pequeña,
              no hay evidencia fuerte de asociación lineal con AUC.
            - N ≈ 20 es **exploratorio**: no generalizar a la población HCP.
            - La edad es el punto medio del rango público, no la edad exacta.
            """
        )

# ===================== MODELO B =====================
elif page == "Modelo B — Demo FC (rojo)":
    style_header_red("Modelo B — Demostración con FC sintética")
    st.warning(
        "**La conectividad funcional de este modelo NO proviene de HCP.** "
        "Se generó de forma sintética para ilustrar el pipeline completo "
        "(grosor + FC) porque el resting-state real (~227 GB) no se descargó."
    )
    st.markdown(
        """
        **Especificación (demo)**

        $$
        \\mathrm{AUC}_i =
        \\beta_0 + \\beta_1\\,\\mathrm{Thickness}_i
        + \\beta_2\\,\\mathrm{FC\\_synthetic}_i
        + \\beta_3\\,\\mathrm{Gender}_i
        + \\beta_4\\,\\mathrm{Age}_i + \\varepsilon_i
        $$
        """
    )

    model, data = fit_model_b(df)
    stats = fit_stats(model)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("N", stats["N"])
    c2.metric("R²", f"{stats['R²']:.3f}")
    c3.metric("R² ajustado", f"{stats['R² ajustado']:.3f}")
    c4.metric("AIC", f"{stats['AIC']:.1f}")

    st.subheader("Coeficientes (no interpretar FC como resultado empírico)")
    st.dataframe(model_table(model).style.format({
        "Coeficiente": "{:.4f}",
        "Error std.": "{:.4f}",
        "t": "{:.3f}",
        "p-valor": "{:.4f}",
    }), use_container_width=True)

    st.subheader("AUC vs FC sintética (solo ilustrativo)")
    fig = px.scatter(
        data,
        x="FC_synthetic",
        y="DDisc_AUC_200",
        color="Thickness_mean",
        hover_data=["Subject"],
        template="plotly_dark",
        labels={
            "FC_synthetic": "FC sintética (z)",
            "DDisc_AUC_200": "AUC $200",
        },
        title="Demo — no usar para conclusiones científicas sobre FC",
    )
    fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    st.plotly_chart(fig, use_container_width=True)

# ===================== GRAFO 3D =====================
elif page == "Grafo 3D":
    st.title("Visualización 3D (estilo oscuro)")
    st.markdown(
        """
        Vista didáctica del cerebro / red regional.  
        El **modelo estadístico** usa el grosor medio tabular por sujeto;
        este gráfico orienta dónde se sitúa la medida cortical.
        """
    )

    use_nilearn = st.checkbox("Intentar superficie nilearn (fsaverage5)", value=True)
    fig = None
    if use_nilearn:
        with st.spinner("Cargando superficie cortical..."):
            fig = try_nilearn_surface_figure(
                "Superficie cortical (fsaverage5) · fondo oscuro"
            )
        if fig is None:
            st.info("No se pudo cargar nilearn/fsaverage; se muestra grafo de nodos.")
    if fig is None:
        fig = brain_graph_dark(
            "Red regional ilustrativa · grosor (escala orientativa)"
        )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Para el grafo con FC real y superficies HCP de un sujeto (p. ej. 100206), "
        "usar el notebook Colab de referencia con CIFTI REST1_LR + midthickness."
    )

# ===================== LIMITACIONES =====================
else:
    st.title("Datos, entrega y limitaciones")
    st.markdown(
        f"""
        ### Archivo de datos
        - Ruta en el repo: `data/subjects_features_real.csv`
        - Filas cargadas: **{len(df)}**
        - Columnas: `{', '.join(df.columns)}`

        ### Origen HCP
        - Conducta: AUC de Delay Discounting (Open Access)
        - Grosor: mapa `thickness_MSMAll.32k_fs_LR.dscalar.nii` (Structural Recommended)
        - Muestra piloto por límite de almacenamiento (sin rfMRI completo)

        ### Qué compartir con el profesor
        1. Link de **Streamlit Cloud** (esta app)
        2. Repo GitHub (código + CSV pequeño)
        3. **No** subir los 30 GB ni paquetes BALSA al repo público

        ### Limitaciones
        - N pequeño → inferencia exploratoria
        - Sin FC real en el Modelo A
        - Modelo B: FC sintética, etiquetada en rojo
        - Edad en rangos, no continua exacta
        """
    )
    st.dataframe(df.describe(include="all"), use_container_width=True)
