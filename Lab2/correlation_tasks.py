"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    df = pd.read_csv(data.csv_path, sep="\t", na_values="NA")

    men = df[df["Gender"] == "Male"]
    women = df[df["Gender"] == "Female"]

    men_count = len(men)
    women_count = len(women)

    men_mri_correlation = {
        feat: float(men[feat].corr(men["MRI_Count"], method="pearson"))
        for feat in FEATURES
    }
    women_mri_correlation = {
        feat: float(women[feat].corr(women["MRI_Count"], method="pearson"))
        for feat in FEATURES
    }

    candidates = [
        (feat, abs(corr))
        for corr_dict in (men_mri_correlation, women_mri_correlation)
        for feat, corr in corr_dict.items()
    ]
    strongest_mri_feature = max(candidates, key=lambda x: x[1])[0]

    return BrainCorrelationSummary(
        men_count=men_count,
        women_count=women_count,
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=strongest_mri_feature,
    )