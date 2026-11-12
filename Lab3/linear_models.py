from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).ravel()
    X_test = np.asarray(data.test_features, dtype=float)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    n, d = X_train_s.shape
    w = np.zeros(d, dtype=float)
    b = 0.0
    lr = data.learning_rate
    epochs = data.epochs
    loss_history = []

    for _ in range(epochs):
        y_pred = X_train_s @ w + b
        error = y_pred - y_train
        mse = float(np.mean(error ** 2))
        loss_history.append(mse)
        grad_w = (2.0 / n) * (X_train_s.T @ error)
        grad_b = (2.0 / n) * np.sum(error)
        w -= lr * grad_w
        b -= lr * grad_b

    predictions = X_test_s @ w + b
    return RegressionResult(
        weights=w,
        bias=float(b),
        loss_history=loss_history,
        predictions=predictions,
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).ravel()
    X_test = np.asarray(data.test_features, dtype=float)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    n, d = X_train_s.shape
    w = np.zeros(d, dtype=float)
    b = 0.0
    lr = data.learning_rate
    epochs = data.epochs
    loss_history = []
    eps = 1e-15

    for _ in range(epochs):
        z = X_train_s @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))
        loss = float(
            -np.mean(
                y_train * np.log(p + eps) + (1.0 - y_train) * np.log(1.0 - p + eps)
            )
        )
        loss_history.append(loss)
        error = p - y_train
        grad_w = (1.0 / n) * (X_train_s.T @ error)
        grad_b = (1.0 / n) * np.sum(error)
        w -= lr * grad_w
        b -= lr * grad_b

    z_test = X_test_s @ w + b
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(z_test, -500.0, 500.0)))
    predictions = (probabilities >= 0.5).astype(int)

    return ClassificationResult(
        weights=w,
        bias=float(b),
        loss_history=loss_history,
        probabilities=probabilities,
        predictions=predictions,
    )


def _compute_metrics(y_true, y_prob, y_pred) -> ClassificationMetrics:
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните вашу логистическую регрессию, DecisionTreeClassifier и KNeighborsClassifier.

    На том же разбиении считаются метрики собственной логистической модели,
    DecisionTree и KNN. Результаты возвращаются в BaselineComparisonResult.
    """
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target).ravel()
    X_test = np.asarray(data.test_features, dtype=float)
    y_test = np.asarray(data.test_target).ravel()

    # 1. Собственная логистическая регрессия (через train_logistic_classifier)
    own_result = train_logistic_classifier(data)
    own_metrics = _compute_metrics(
        y_test, own_result.probabilities, own_result.predictions
    )

    # 2. Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_prob = dt.predict_proba(X_test)[:, 1]
    dt_pred = dt.predict(X_test)
    dt_metrics = _compute_metrics(y_test, dt_prob, dt_pred)

    # 3. KNN
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    knn = KNeighborsClassifier()
    knn.fit(X_train_s, y_train)
    knn_prob = knn.predict_proba(X_test_s)[:, 1]
    knn_pred = knn.predict(X_test_s)
    knn_metrics = _compute_metrics(y_test, knn_prob, knn_pred)

    return BaselineComparisonResult(
        logistic_regression=own_metrics,  # собственная модель
        decision_tree=dt_metrics,
        knn=knn_metrics,
    )
