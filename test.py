from __future__ import annotations

from pathlib import Path
from pickle import dump

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TARGET_COLUMN = "購入フラグ"
ID_COLUMN = "企業ID"
MODEL_DIR = OUTPUT_DIR / "models"


def build_preprocessor(train_frame: pd.DataFrame) -> ColumnTransformer:
    feature_frame = train_frame.drop(columns=[TARGET_COLUMN])

    numeric_columns = feature_frame.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column
        for column in ["業界", "上場種別", "特徴"]
        if column in feature_frame.columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def build_pipeline(train_frame: pd.DataFrame) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(train_frame)),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample_submit = pd.read_csv(DATA_DIR / "sample_submit.csv", header=None)

    y = train[TARGET_COLUMN]
    X = train.drop(columns=[TARGET_COLUMN])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    test_pred_proba = pd.Series(0.0, index=test.index)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for fold_index, (train_index, valid_index) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_index]
        X_valid = X.iloc[valid_index]
        y_train = y.iloc[train_index]
        y_valid = y.iloc[valid_index]

        pipeline = build_pipeline(train)

        pipeline.fit(X_train, y_train)

        valid_pred = pipeline.predict(X_valid)
        fold_f1 = f1_score(y_valid, valid_pred)
        cv_scores.append(fold_f1)
        print(f"Fold {fold_index} F1: {fold_f1:.4f}")

        fold_test_proba = pipeline.predict_proba(test)[:, 1]
        test_pred_proba += pd.Series(fold_test_proba, index=test.index)

        model_path = MODEL_DIR / f"fold_{fold_index}.pkl"
        with model_path.open("wb") as file_handle:
            dump(pipeline, file_handle)
        print(f"Saved fold {fold_index} model to {model_path}")

    test_pred_proba /= cv.get_n_splits()
    test_pred = (test_pred_proba >= 0.5).astype(int)
    print(f"CV Mean F1: {sum(cv_scores) / len(cv_scores):.4f}")

    sample_submit.iloc[:, 1] = test_pred
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = OUTPUT_DIR / "submission.csv"
    sample_submit.to_csv(submission_path, index=False, header=None)

    print(f"Saved submission to {submission_path}")


if __name__ == "__main__":
    main()