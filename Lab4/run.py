import pandas as pd
from sklearn.model_selection import train_test_split
from ensemble_models import run_ensemble_experiment
from grader_contracts.ensembles import EnsembleInput

df = pd.read_csv("german.csv", sep=";")
X = df.drop("Creditability", axis=1)
y = df["Creditability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

data = EnsembleInput(
    train_features=X_train,
    train_target=y_train,
    test_features=X_test,
    test_target=y_test,
    random_seed=42,
)

result = run_ensemble_experiment(data)

print("Модель:", result.model_name)
print(f"ROC-AUC:   {result.roc_auc:.4f}")
print(f"Accuracy:  {result.accuracy:.4f}")
print(f"Precision: {result.precision:.4f}")
print(f"Recall:    {result.recall:.4f}")
print(f"F1:        {result.f1:.4f}")