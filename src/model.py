"""Regresiones del Modelo A (real) y Modelo B (demo FC sintética)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def prepare_dataframe(df: pd.DataFrame, add_synthetic_fc: bool = False, seed: int = 2026) -> pd.DataFrame:
    """Limpia columnas y opcionalmente añade FC sintética correlacionada débilmente con DD."""
    out = df.copy()
    out["Subject"] = out["Subject"].astype(int)

    # Gender numérico
    out["Gender_M"] = out["Gender"].astype(str).str.upper().map({"M": 1, "F": 0})

    # Edad: punto medio del rango público HCP
    def age_mid(x):
        s = str(x).strip()
        if "-" in s or "–" in s:
            parts = s.replace("–", "-").split("-")
            try:
                return (float(parts[0]) + float(parts[1])) / 2.0
            except Exception:
                return np.nan
        if "+" in s:
            try:
                return float(s.replace("+", "")) + 2.5
            except Exception:
                return np.nan
        try:
            return float(s)
        except Exception:
            return np.nan

    out["Age_mid"] = out["Age"].map(age_mid)

    out = out.dropna(subset=["DDisc_AUC_200", "Thickness_mean", "Gender_M", "Age_mid"])

    if add_synthetic_fc:
        rng = np.random.default_rng(seed)
        # FC sintética: ruido + pequeña relación con thickness y con DD (solo demo)
        out = out.copy()
        out["FC_synthetic"] = (
            0.15 * (out["Thickness_mean"] - out["Thickness_mean"].mean())
            + 0.10 * (out["DDisc_AUC_200"] - out["DDisc_AUC_200"].mean())
            + rng.normal(0, 0.25, size=len(out))
        )
        # estandarizar a media 0 para lectura fácil
        out["FC_synthetic"] = (
            out["FC_synthetic"] - out["FC_synthetic"].mean()
        ) / (out["FC_synthetic"].std(ddof=0) + 1e-12)

    return out.reset_index(drop=True)


def fit_model_a(df: pd.DataFrame):
    """Modelo A real: AUC_200 ~ Thickness + Gender + Age."""
    data = prepare_dataframe(df, add_synthetic_fc=False)
    model = smf.ols(
        "DDisc_AUC_200 ~ Thickness_mean + Gender_M + Age_mid",
        data=data,
    ).fit()
    return model, data


def fit_model_b(df: pd.DataFrame, seed: int = 2026):
    """Modelo B demo: AUC_200 ~ Thickness + FC_synthetic + Gender + Age."""
    data = prepare_dataframe(df, add_synthetic_fc=True, seed=seed)
    model = smf.ols(
        "DDisc_AUC_200 ~ Thickness_mean + FC_synthetic + Gender_M + Age_mid",
        data=data,
    ).fit()
    return model, data


def model_table(model) -> pd.DataFrame:
    """Tabla legible de coeficientes."""
    rows = []
    for name in model.params.index:
        rows.append(
            {
                "Variable": name,
                "Coeficiente": float(model.params[name]),
                "Error std.": float(model.bse[name]),
                "t": float(model.tvalues[name]),
                "p-valor": float(model.pvalues[name]),
            }
        )
    return pd.DataFrame(rows)


def fit_stats(model) -> dict:
    return {
        "N": int(model.nobs),
        "R²": float(model.rsquared),
        "R² ajustado": float(model.rsquared_adj),
        "F": float(model.fvalue) if model.fvalue is not None else np.nan,
        "p (F)": float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
        "AIC": float(model.aic),
    }
