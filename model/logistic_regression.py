from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml_pipeline_utils import (
    RANDOM_STATE,
    load_data,
    build_preprocessor,
    split_data,
    evaluate_model,
)


def main():
    _, X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = evaluate_model(
        "Logistic Regression", y_test, y_pred, y_prob
    )

    print("\nLogistic Regression Results")
    print("-" * 35)
    for k, v in results.items():
        print(f"{k}: {v:.4f}" if k != "ML Model Name" else f"{k}: {v}")


if __name__ == "__main__":
    main()
