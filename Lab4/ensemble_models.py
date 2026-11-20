"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучите выбранную модель и верните вероятности, метки и метрики на test.

    Разрешены Random Forest, Extra Trees, AdaBoost, Gradient Boosting,
    HistGradientBoosting, Voting/Stacking или полносвязная MLP. Не используйте
    test_target при обучении модели или подборе её параметров.
    """
    X_train = np.asarray(data.train_features)
    y_train = np.asarray(data.train_target).ravel()
    X_test = np.asarray(data.test_features)
    y_test = np.asarray(data.test_target).ravel()
    seed = data.random_seed

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed + 1,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=seed + 2,
    )

    model = VotingClassifier(
        estimators=[("rf", rf), ("et", et), ("gb", gb)],
        voting="soft",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    roc_auc = float(roc_auc_score(y_test, probabilities))
    accuracy = float(accuracy_score(y_test, predictions))
    precision = float(precision_score(y_test, predictions, zero_division=0))
    recall = float(recall_score(y_test, predictions, zero_division=0))
    f1 = float(f1_score(y_test, predictions, zero_division=0))

    return EnsembleResult(
        model_name="VotingClassifier(RF+ET+GB, soft)",
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=roc_auc,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )
