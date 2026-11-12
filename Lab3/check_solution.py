"""Проверка работоспособности решения лабораторной №3."""

from linear_models import (
    train_linear_regression,
    train_logistic_classifier,
    compare_classical_models,
)
from grader_contracts.linear_models import RegressionInput, ClassificationInput
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score


def main():
    print("Импорт OK\n")

    # 1. Линейная регрессия
    print("=== 1. Линейная регрессия ===")
    df = pd.read_csv("Student_Performance.csv")
    df["Extracurricular Activities"] = (df["Extracurricular Activities"] == "Yes").astype(int)
    X = df.drop("Performance Index", axis=1).values.astype(float)
    y = df["Performance Index"].values.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    reg_in = RegressionInput(X_train, y_train, X_test, y_test)
    reg_out = train_linear_regression(reg_in)
    mse = mean_squared_error(y_test, reg_out.predictions)
    print(f"  test MSE:          {mse:.4f}  (ожидается ≈ 4.08)")
    print(f"  loss_history len:  {len(reg_out.loss_history)}")
    print(f"  final train loss:  {reg_out.loss_history[-1]:.4f}")
    print(f"  bias:              {reg_out.bias:.4f}")
    print(f"  weights shape:     {np.asarray(reg_out.weights).shape}")
    print()

    # 2. Логистическая регрессия
    print("=== 2. Собственный логистический классификатор ===")
    df = pd.read_csv("german.csv", sep=";")
    y = df["Creditability"].values
    X = df.drop("Creditability", axis=1).values.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf_in = ClassificationInput(X_train, y_train, X_test, y_test)
    clf_out = train_logistic_classifier(clf_in)
    auc = roc_auc_score(y_test, clf_out.probabilities)
    acc = accuracy_score(y_test, clf_out.predictions)
    print(f"  ROC-AUC:           {auc:.4f}  (ожидается ≈ 0.808)")
    print(f"  Accuracy:          {acc:.4f}  (ожидается ≈ 0.77)")
    print(f"  loss_history len:  {len(clf_out.loss_history)}")
    print(f"  final train loss:  {clf_out.loss_history[-1]:.4f}")
    print(f"  probabilities shape: {np.asarray(clf_out.probabilities).shape}")
    print(f"  predictions shape:   {np.asarray(clf_out.predictions).shape}")
    print()

    # 3. Сравнение моделей (own logistic + DT + KNN)
    print("=== 3. Сравнение: своя логистика vs DecisionTree vs KNN ===")
    comp = compare_classical_models(clf_in)
    print(f"  Own Logistic      — ROC-AUC: {comp.logistic_regression.roc_auc:.4f}, "
          f"Acc: {comp.logistic_regression.accuracy:.4f}")
    print(f"  Decision Tree     — ROC-AUC: {comp.decision_tree.roc_auc:.4f}, "
          f"Acc: {comp.decision_tree.accuracy:.4f}")
    print(f"  KNN               — ROC-AUC: {comp.knn.roc_auc:.4f}, "
          f"Acc: {comp.knn.accuracy:.4f}")
    print()


if __name__ == "__main__":
    main()
