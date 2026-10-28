# Delay Discounting ~ Grosor cortical (HCP-YA) + demo con FC sintética

App Streamlit para entregar al profesor: **datos reales HCP** (muestra piloto) + visualización 3D.

## Modelos

| Modelo | Color | Datos | Descripción |
|--------|-------|--------|-------------|
| **A — Real** | Verde | 21 sujetos HCP-YA | `DDisc_AUC_200 ~ Thickness_mean + Gender + Age` |
| **B — Demo FC** | Rojo | Mismos + **FC sintética** | Ilustra el pipeline completo; la FC **no** es de HCP |

### Por qué no hay FC real

El resting-state preprocesado de ~21 sujetos ocupa ~227 GB. Con límite de disco se descargó solo **Structural Preprocessed Recommended** (~30 GB) para extraer grosor cortical. El Modelo B añade FC simulada solo con fines metodológicos.

## Estructura del repo

```text
hcp_dd_streamlit/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── subjects_features_real.csv   ← súbelo tú (no subir los .nii / zips HCP)
└── src/
    ├── model.py
    └── brain_plot.py
```

## Cómo publicar en Streamlit Cloud

1. Crea un repo en GitHub y sube esta carpeta con **GitHub Desktop**.
2. Coloca tu `subjects_features_real.csv` en `data/`.
3. En [share.streamlit.io](https://share.streamlit.io): New app → elige el repo → Main file: `app.py`.
4. Envía al profesor solo el **link** de la app.

## Reproducibilidad local

```bash
cd hcp_dd_streamlit
pip install -r requirements.txt
streamlit run app.py
```

## Datos HCP

- Release: HCP-YA 2025 (ConnectomeDB powered by BALSA)
- Paquete: Structural Preprocessed Recommended (thickness MSMAll 32k)
- Conducta: `DDisc_AUC_200`, `DDisc_AUC_40K`
- Cada usuario de datos HCP debe aceptar los Data Use Terms en ConnectomeDB/BALSA.

## Limitaciones

- N ≈ 21: resultados **exploratorios / piloto**.
- Edad en rangos públicos (no edad exacta).
- No se publican paquetes de imagen originales en el repo.
