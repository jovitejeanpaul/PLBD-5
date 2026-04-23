"""
data_processing.py
==================
Pipeline de traitement des données pour la prédiction de la potabilité de l'eau.

Dataset source : https://www.kaggle.com/datasets/adityakadiwal/water-potability
Features utilisées : Conductivity, Solids, Turbidity, ph
Variable cible     : Potability (0 = non potable, 1 = potable)

Modules
-------
- optimize_memory       : Réduction de l'empreinte mémoire par downcasting des types.
- load_and_select       : Chargement du CSV et sélection des colonnes pertinentes.
- describe_missing      : Rapport synthétique sur les valeurs manquantes.
- detect_outliers_iqr   : Détection des valeurs aberrantes via la méthode IQR.
- cap_outliers_iqr      : Traitement des outliers par écrêtage (winsorisation).
- impute_conductivity   : Imputation de Conductivity à partir de Solids (TDS proxy).
- impute_solids         : Imputation de Solids à partir de Conductivity.
- impute_ph_by_group    : Imputation de ph par médiane conditionnelle (groupe Potability).
- impute_turbidity      : Imputation de Turbidity par médiane globale.
- run_pipeline          : Orchestration complète du pipeline.

Dépendances
-----------
    pip install pandas numpy scikit-learn scipy
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------------------------
# Configuration du logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Colonnes nécessaires à la modélisation + cible
FEATURES = ["ph", "Solids", "Conductivity", "Turbidity"]
TARGET = "Potability"
ALL_COLS = FEATURES + [TARGET]

# Bornes physiques raisonnables pour chaque feature (domaine eau potable/naturelle)
PHYSICAL_BOUNDS: dict[str, Tuple[float, float]] = {
    "ph":           (0.0,  14.0),
    "Solids":       (0.0,  70_000.0),   # mg/L — TDS eau douce / saumâtre
    "Conductivity": (0.0,  1_500.0),    # µS/cm — eau douce à légèrement minéralisée
    "Turbidity":    (0.0,  100.0),      # NTU
}

# Relation empirique TDS ↔ Conductivity :  TDS (mg/L) ≈ k × EC (µS/cm)
# Le facteur k varie de 0.55 à 0.75 selon la composition ionique ;
# 0.64 est la valeur standard retenue par l'OMS pour l'eau potable.
TDS_EC_FACTOR: float = 0.64


# ===========================================================================
# 1. OPTIMISATION MÉMOIRE
# ===========================================================================

def optimize_memory(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Réduit l'empreinte mémoire d'un DataFrame par downcasting des types numériques.

    Stratégie appliquée :
    - Colonnes entières  → plus petit type int (int8 … int64)
    - Colonnes flottantes → float32 (suffisant pour la précision requise)
    - La colonne cible binaire (0/1) est convertie en int8

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame source.
    verbose : bool, optional
        Si True, affiche le gain mémoire obtenu. Par défaut True.

    Returns
    -------
    pd.DataFrame
        DataFrame avec types optimisés (copie indépendante).

    Examples
    --------
    >>> df_opt = optimize_memory(df)
    Mémoire avant : 2.45 MB  →  après : 0.61 MB  (gain : 75.1 %)
    """
    df = df.copy()
    mem_before = df.memory_usage(deep=True).sum() / 1024 ** 2  # MB

    for col in df.select_dtypes(include=["integer"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in df.select_dtypes(include=["float"]).columns:
        df[col] = df[col].astype(np.float32)

    # Cible binaire → int8
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].astype(np.int8)

    mem_after = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        gain = (1 - mem_after / mem_before) * 100 if mem_before > 0 else 0
        logger.info(
            "Mémoire avant : %.2f MB  →  après : %.2f MB  (gain : %.1f %%)",
            mem_before, mem_after, gain,
        )
    return df


# ===========================================================================
# 2. CHARGEMENT & SÉLECTION DES COLONNES
# ===========================================================================

def load_and_select(
    filepath: str | Path,
    optimize: bool = True,
) -> pd.DataFrame:
    """
    Charge le CSV Kaggle et retourne uniquement les colonnes nécessaires.

    Parameters
    ----------
    filepath : str | Path
        Chemin vers le fichier ``water_potability.csv``.
    optimize : bool, optional
        Si True, applique :func:`optimize_memory` après le chargement. Par défaut True.

    Returns
    -------
    pd.DataFrame
        DataFrame filtré sur ``ALL_COLS`` = FEATURES + TARGET.

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas au chemin spécifié.
    KeyError
        Si des colonnes attendues sont absentes du CSV.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")

    df = pd.read_csv(filepath)
    logger.info("Dataset chargé : %d lignes × %d colonnes", *df.shape)

    missing_cols = set(ALL_COLS) - set(df.columns)
    if missing_cols:
        raise KeyError(f"Colonnes manquantes dans le CSV : {missing_cols}")

    df = df[ALL_COLS].copy()
    logger.info("Colonnes sélectionnées : %s", ALL_COLS)

    if optimize:
        df = optimize_memory(df)

    return df


# ===========================================================================
# 3. DIAGNOSTIC DES VALEURS MANQUANTES
# ===========================================================================

def describe_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Génère un rapport sur les valeurs manquantes par colonne.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à analyser.

    Returns
    -------
    pd.DataFrame
        Tableau avec les colonnes ``n_missing``, ``pct_missing`` et ``dtype``
        pour chaque variable présentant au moins une valeur manquante.

    Examples
    --------
    >>> report = describe_missing(df)
    >>> print(report)
    """
    missing = df.isnull().sum()
    pct = missing / len(df) * 100
    report = (
        pd.DataFrame({"n_missing": missing, "pct_missing": pct.round(2), "dtype": df.dtypes})
        .query("n_missing > 0")
        .sort_values("pct_missing", ascending=False)
    )
    logger.info("Valeurs manquantes :\n%s", report.to_string())
    return report


# ===========================================================================
# 4. DÉTECTION & TRAITEMENT DES VALEURS ABERRANTES
# ===========================================================================

def detect_outliers_iqr(
    df: pd.DataFrame,
    cols: Optional[list[str]] = None,
    factor: float = 1.5,
) -> pd.DataFrame:
    """
    Identifie les outliers via la règle IQR (méthode de Tukey).

    Un point *x* est considéré aberrant si :
        x < Q1 - factor × IQR  ou  x > Q3 + factor × IQR

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame source.
    cols : list[str], optional
        Liste des colonnes à analyser. Par défaut : FEATURES.
    factor : float, optional
        Multiplicateur IQR. 1.5 = outliers modérés, 3.0 = outliers extrêmes.
        Par défaut 1.5.

    Returns
    -------
    pd.DataFrame
        Rapport avec colonnes ``Q1``, ``Q3``, ``IQR``, ``lower``, ``upper``,
        ``n_outliers``, ``pct_outliers`` pour chaque colonne analysée.
    """
    cols = cols or FEATURES
    records = []

    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        n_out = ((s < lower) | (s > upper)).sum()
        records.append({
            "column":     col,
            "Q1":         round(float(q1), 4),
            "Q3":         round(float(q3), 4),
            "IQR":        round(float(iqr), 4),
            "lower":      round(float(lower), 4),
            "upper":      round(float(upper), 4),
            "n_outliers": int(n_out),
            "pct_outliers": round(n_out / len(s) * 100, 2),
        })

    report = pd.DataFrame(records).set_index("column")
    logger.info("Rapport outliers IQR (factor=%.1f) :\n%s", factor, report.to_string())
    return report


def cap_outliers_iqr(
    df: pd.DataFrame,
    cols: Optional[list[str]] = None,
    factor: float = 1.5,
    enforce_physical: bool = True,
) -> pd.DataFrame:
    """
    Écrête (winssorise) les valeurs aberrantes aux bornes IQR.

    Pour chaque colonne, les valeurs inférieures à ``lower`` sont remplacées
    par ``lower`` et celles supérieures à ``upper`` par ``upper``.
    Optionnellement, les bornes physiques de :data:`PHYSICAL_BOUNDS` sont
    également appliquées pour éviter des valeurs physiquement impossibles.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame source.
    cols : list[str], optional
        Colonnes à traiter. Par défaut : FEATURES.
    factor : float, optional
        Multiplicateur IQR. Par défaut 1.5.
    enforce_physical : bool, optional
        Si True, clip également selon :data:`PHYSICAL_BOUNDS`. Par défaut True.

    Returns
    -------
    pd.DataFrame
        DataFrame avec outliers écrêtés (copie indépendante).
    """
    df = df.copy()
    cols = cols or FEATURES

    for col in cols:
        if col not in df.columns:
            continue

        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr

        # Appliquer les bornes physiques si demandé
        if enforce_physical and col in PHYSICAL_BOUNDS:
            phys_low, phys_high = PHYSICAL_BOUNDS[col]
            lower = max(lower, phys_low)
            upper = min(upper, phys_high)

        before = df[col].copy()
        df[col] = df[col].clip(lower=lower, upper=upper)
        n_capped = (df[col] != before).sum()
        logger.info("  [%s] %d valeurs écrêtées → [%.3f, %.3f]", col, n_capped, lower, upper)

    return df


# ===========================================================================
# 5. IMPUTATION PAR CORRÉLATION
# ===========================================================================

def impute_conductivity(
    df: pd.DataFrame,
    factor: float = TDS_EC_FACTOR,
) -> pd.DataFrame:
    """
    Impute les valeurs manquantes de ``Conductivity`` à partir de ``Solids``.

    Relation physique utilisée (loi empirique TDS–EC) :
        Conductivity (µS/cm) = Solids (mg/L) / factor

    avec ``factor`` ≈ 0.64 (valeur OMS standard pour eau potable).

    Cette imputation n'est appliquée qu'aux lignes où ``Conductivity`` est
    manquante **et** ``Solids`` est disponible.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant au moins les colonnes ``Conductivity`` et ``Solids``.
    factor : float, optional
        Facteur de conversion TDS → EC. Par défaut :data:`TDS_EC_FACTOR` = 0.64.

    Returns
    -------
    pd.DataFrame
        DataFrame avec ``Conductivity`` imputée (copie indépendante).

    References
    ----------
    - WHO (2017). *Guidelines for Drinking-water Quality*, 4th ed., p. 218.
    - APHA (2017). *Standard Methods for the Examination of Water and Wastewater*.
    """
    df = df.copy()
    mask = df["Conductivity"].isna() & df["Solids"].notna()
    n = mask.sum()
    if n > 0:
        df.loc[mask, "Conductivity"] = df.loc[mask, "Solids"] / factor
        logger.info("Conductivity : %d valeurs imputées via Solids / %.2f", n, factor)
    else:
        logger.info("Conductivity : aucune imputation nécessaire via Solids.")
    return df


def impute_solids(
    df: pd.DataFrame,
    factor: float = TDS_EC_FACTOR,
) -> pd.DataFrame:
    """
    Impute les valeurs manquantes de ``Solids`` (TDS) à partir de ``Conductivity``.

    Relation physique utilisée :
        Solids (mg/L) = factor × Conductivity (µS/cm)

    Cette imputation n'est appliquée qu'aux lignes où ``Solids`` est manquant
    **et** ``Conductivity`` est disponible.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant au moins les colonnes ``Conductivity`` et ``Solids``.
    factor : float, optional
        Facteur de conversion EC → TDS. Par défaut :data:`TDS_EC_FACTOR` = 0.64.

    Returns
    -------
    pd.DataFrame
        DataFrame avec ``Solids`` imputée (copie indépendante).
    """
    df = df.copy()
    mask = df["Solids"].isna() & df["Conductivity"].notna()
    n = mask.sum()
    if n > 0:
        df.loc[mask, "Solids"] = df.loc[mask, "Conductivity"] * factor
        logger.info("Solids : %d valeurs imputées via Conductivity × %.2f", n, factor)
    else:
        logger.info("Solids : aucune imputation nécessaire via Conductivity.")
    return df


def impute_ph_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute les valeurs manquantes de ``ph`` par médiane conditionnelle au groupe ``Potability``.

    Rationale : le pH des eaux potables est généralement plus resserré (6.5–8.5
    selon l'OMS) que celui des eaux non potables. Utiliser la médiane par classe
    préserve cette distribution bimodale potentielle.

    Si ``Potability`` n'est pas disponible, la médiane globale est utilisée
    comme fallback.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant ``ph`` et éventuellement ``Potability``.

    Returns
    -------
    pd.DataFrame
        DataFrame avec ``ph`` imputée (copie indépendante).
    """
    df = df.copy()
    mask = df["ph"].isna()
    n_missing = mask.sum()

    if n_missing == 0:
        logger.info("ph : aucune valeur manquante.")
        return df

    if TARGET in df.columns and df[TARGET].notna().any():
        # Imputation par médiane de groupe
        group_medians = df.groupby(TARGET)["ph"].median()
        for grp, med in group_medians.items():
            idx = mask & (df[TARGET] == grp)
            df.loc[idx, "ph"] = med
            logger.info("  ph groupe %s : %d valeurs imputées (médiane = %.4f)", grp, idx.sum(), med)

        # Fallback pour les lignes sans Potability connue
        remaining = df["ph"].isna()
        if remaining.any():
            global_median = df["ph"].median()
            df.loc[remaining, "ph"] = global_median
            logger.info("  ph fallback global : %d valeurs imputées (médiane = %.4f)", remaining.sum(), global_median)
    else:
        global_median = df["ph"].median()
        df["ph"] = df["ph"].fillna(global_median)
        logger.info("ph : %d valeurs imputées par médiane globale (%.4f)", n_missing, global_median)

    return df


def impute_turbidity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute les valeurs manquantes de ``Turbidity`` par médiane globale.

    La turbidité n'a pas de corrélation physique forte et directe avec les
    autres features retenues ; la médiane est robuste aux outliers résiduels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant la colonne ``Turbidity``.

    Returns
    -------
    pd.DataFrame
        DataFrame avec ``Turbidity`` imputée (copie indépendante).
    """
    df = df.copy()
    n_missing = df["Turbidity"].isna().sum()
    if n_missing > 0:
        median_val = df["Turbidity"].median()
        df["Turbidity"] = df["Turbidity"].fillna(median_val)
        logger.info("Turbidity : %d valeurs imputées (médiane = %.4f)", n_missing, median_val)
    else:
        logger.info("Turbidity : aucune valeur manquante.")
    return df


# ===========================================================================
# 6. VÉRIFICATION FINALE
# ===========================================================================

def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Vérifie l'intégrité du DataFrame après traitement.

    Contrôles effectués :
    - Absence de valeurs manquantes dans les features et la cible.
    - Respect des bornes physiques de :data:`PHYSICAL_BOUNDS`.
    - Distribution de la cible (équilibre des classes).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame traité à valider.

    Raises
    ------
    ValueError
        Si des valeurs manquantes persistent ou si des bornes physiques
        sont violées après traitement.
    """
    # Valeurs manquantes résiduelles
    remaining_na = df[ALL_COLS].isnull().sum()
    if remaining_na.any():
        raise ValueError(
            f"Valeurs manquantes résiduelles après pipeline :\n{remaining_na[remaining_na > 0]}"
        )

    # Bornes physiques
    violations = []
    for col, (low, high) in PHYSICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        out_of_bounds = ((df[col] < low) | (df[col] > high)).sum()
        if out_of_bounds:
            violations.append(f"  {col}: {out_of_bounds} valeurs hors [{low}, {high}]")

    if violations:
        raise ValueError("Violations des bornes physiques :\n" + "\n".join(violations))

    # Distribution de la cible
    target_dist = df[TARGET].value_counts(normalize=True).round(3)
    logger.info("Distribution de %s :\n%s", TARGET, target_dist.to_string())
    logger.info("✓ Validation réussie. Shape final : %s", df.shape)


# ===========================================================================
# 7. PIPELINE PRINCIPAL
# ===========================================================================

def run_pipeline(
    filepath: str | Path,
    cap_factor: float = 1.5,
    tds_ec_factor: float = TDS_EC_FACTOR,
    return_X_y: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.Series]:
    """
    Orchestre l'ensemble du pipeline de traitement des données.

    Étapes :
    1. Chargement & sélection des colonnes (+ optimisation mémoire)
    2. Rapport sur les valeurs manquantes
    3. Détection des outliers (rapport uniquement)
    4. Écrêtage des outliers IQR + respect des bornes physiques
    5. Imputation Conductivity ← Solids (relation TDS–EC)
    6. Imputation Solids ← Conductivity (relation EC–TDS)
    7. Imputation pH par médiane conditionnelle au groupe Potability
    8. Imputation Turbidity par médiane globale
    9. Validation finale du DataFrame

    Parameters
    ----------
    filepath : str | Path
        Chemin vers ``water_potability.csv``.
    cap_factor : float, optional
        Facteur IQR pour l'écrêtage. Par défaut 1.5.
    tds_ec_factor : float, optional
        Facteur de conversion TDS ↔ EC. Par défaut :data:`TDS_EC_FACTOR`.
    return_X_y : bool, optional
        Si True, retourne le tuple ``(X, y)`` prêt pour scikit-learn.
        Si False (défaut), retourne le DataFrame complet.

    Returns
    -------
    pd.DataFrame
        DataFrame traité, ou tuple ``(X, y)`` si ``return_X_y=True``.

    Examples
    --------
    >>> df = run_pipeline("water_potability.csv")
    >>> X, y = run_pipeline("water_potability.csv", return_X_y=True)
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> clf = RandomForestClassifier().fit(X, y)
    """
    logger.info("=" * 60)
    logger.info("DÉBUT DU PIPELINE — Water Potability")
    logger.info("=" * 60)

    # Étape 1 : Chargement
    df = load_and_select(filepath, optimize=True)

    # Étape 2 : Diagnostic NaN
    describe_missing(df)

    # Étape 3 : Rapport outliers (sans modification)
    detect_outliers_iqr(df, factor=cap_factor)

    # Étape 4 : Écrêtage outliers
    logger.info("--- Écrêtage des outliers (IQR factor=%.1f) ---", cap_factor)
    df = cap_outliers_iqr(df, factor=cap_factor, enforce_physical=True)

    # Étape 5 & 6 : Imputation croisée Conductivity ↔ Solids
    logger.info("--- Imputation croisée Conductivity ↔ Solids ---")
    df = impute_conductivity(df, factor=tds_ec_factor)
    df = impute_solids(df, factor=tds_ec_factor)

    # Étape 7 : Imputation pH
    logger.info("--- Imputation pH par groupe Potability ---")
    df = impute_ph_by_group(df)

    # Étape 8 : Imputation Turbidity
    logger.info("--- Imputation Turbidity ---")
    df = impute_turbidity(df)

    # Étape 9 : Validation
    logger.info("--- Validation ---")
    validate_dataframe(df)

    logger.info("=" * 60)
    logger.info("PIPELINE TERMINÉ")
    logger.info("=" * 60)

    if return_X_y:
        X = df[FEATURES]
        y = df[TARGET]
        return X, y

    return df


# ===========================================================================
# Point d'entrée (usage direct)
# ===========================================================================

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "water_potability.csv"

    X, y = run_pipeline(path, return_X_y=True)

    print("\n--- Aperçu des features (5 premières lignes) ---")
    print(X.head().to_string())
    print(f"\nShape X : {X.shape}  |  Shape y : {y.shape}")
    print(f"Classes y : {dict(y.value_counts().sort_index())}")
