from pathlib import Path
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from flaml import AutoML
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
CSV_DIR = Path("csv")
EXPERIMENT_DIR = Path("experiment")
EXPERIMENT_DIR.mkdir(exist_ok=True)

final_dataset = pd.read_csv(CSV_DIR / "final_dataset_descriptors.csv")
descriptor_cols = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "HeavyAtomCount",
    "FractionCSP3",
    "NumAromaticRings",
]
fingerprint_cols = [c for c in final_dataset.columns if c.startswith("ECFP4_")]
feature_sets = {
    "fingerprint": fingerprint_cols,
    "combined": descriptor_cols + fingerprint_cols,
}

y = final_dataset["label"].astype(int)
X_train_full, X_test_full, y_train, y_test = train_test_split(
    final_dataset.drop(columns=["label"]),
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

baseline = pd.read_csv(CSV_DIR / "model_results.csv")
baseline_score_col = "roc_auc" if "roc_auc" in baseline.columns else "cv_roc_auc"
baseline_metric_map = {
    "cv_accuracy": "accuracy",
    "cv_balanced_accuracy": "balanced_accuracy",
    "cv_f1": "f1",
    "cv_roc_auc": "roc_auc",
}
baseline_for_compare = baseline.rename(columns=baseline_metric_map).copy()
baseline_best = baseline_for_compare.sort_values("roc_auc", ascending=False).iloc[0]
rows = []
best_artifact = {
    "roc_auc": -np.inf,
    "pipeline": None,
    "feature_set": None,
    "model_name": None,
    "columns": None,
}


def evaluate_pipeline(model_name, feature_set_name, pipe):
    cols = feature_sets[feature_set_name]
    pipe.fit(X_train_full[cols], y_train)
    pred = pipe.predict(X_test_full[cols])
    score = pipe.predict_proba(X_test_full[cols])[:, 1]
    row = {
        "feature_set": feature_set_name,
        "model": model_name,
        "n_features": len(cols),
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, score),
    }
    rows.append(row)
    if row["roc_auc"] > best_artifact["roc_auc"]:
        best_artifact.update(
            {
                "roc_auc": row["roc_auc"],
                "pipeline": pipe,
                "feature_set": feature_set_name,
                "model_name": model_name,
                "columns": cols,
            }
        )


for feature_set_name, cols in feature_sets.items():
    automl = AutoML()
    automl.fit(
        X_train_full[cols],
        y_train,
        task="classification",
        metric="roc_auc",
        estimator_list=["rf", "extra_tree", "lrl1", "lrl2"],
        time_budget=20,
        seed=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
    )
    flaml_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", automl.model.estimator),
        ]
    )
    evaluate_pipeline(f"flaml_{automl.best_estimator}", feature_set_name, flaml_pipe)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def tune_random_forest(feature_set_name):
    cols = feature_sets[feature_set_name]
    X_train = X_train_full[cols]

    def objective(trial):
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 300, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 18),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 12),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 6),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            class_weight=trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
        return cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1).mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=8, show_progress_bar=False)
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(**study.best_params, random_state=RANDOM_STATE, n_jobs=1)),
        ]
    )
    evaluate_pipeline("optuna_random_forest", feature_set_name, pipe)


def tune_gradient_boosting(feature_set_name):
    cols = feature_sets[feature_set_name]
    X_train = X_train_full[cols]

    def objective(trial):
        model = GradientBoostingClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 200, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 1, 3),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 12),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 6),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            random_state=RANDOM_STATE,
        )
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
        return cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1).mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=8, show_progress_bar=False)
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(**study.best_params, random_state=RANDOM_STATE)),
        ]
    )
    evaluate_pipeline("optuna_gradient_boosting", feature_set_name, pipe)


for feature_set_name in feature_sets:
    tune_random_forest(feature_set_name)
    tune_gradient_boosting(feature_set_name)

automl_results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
automl_results.to_csv(CSV_DIR / "automl_model_results.csv", index=False)

combined_results = (
    pd.concat([baseline_for_compare, automl_results], ignore_index=True)
    .sort_values("roc_auc", ascending=False)
    .reset_index(drop=True)
)
combined_results.to_csv(EXPERIMENT_DIR / "automl_expanded_experiment_results.csv", index=False)

if best_artifact["roc_auc"] > float(baseline_best["roc_auc"]):
    joblib.dump(best_artifact, EXPERIMENT_DIR / "best_automl_model.joblib")

print("Baseline best")
print(baseline_best.to_dict())
print("\nAutoML results")
print(automl_results.to_string(index=False))
print("\nCombined top 10")
print(combined_results.head(10).to_string(index=False))
print(
    "\nImproved:",
    bool(best_artifact["roc_auc"] > float(baseline_best["roc_auc"])),
    "automl_best_roc_auc=",
    round(best_artifact["roc_auc"], 6),
    "baseline_best_roc_auc=",
    round(float(baseline_best["roc_auc"]), 6),
)
