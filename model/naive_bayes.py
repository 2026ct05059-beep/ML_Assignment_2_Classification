from sklearn.naive_bayes import GaussianNB

from ml_pipeline_utils import (
    load_data,
    build_preprocessor,
    split_data,
    evaluate_model,
)


def main():
    _, X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Fit preprocessing only on the training data.
    preprocessor = build_preprocessor(X)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model = GaussianNB()
    model.fit(X_train_processed, y_train)

    y_pred = model.predict(X_test_processed)
    y_prob = model.predict_proba(X_test_processed)[:, 1]

    results = evaluate_model(
        "Gaussian Naive Bayes", y_test, y_pred, y_prob
    )

    print("\nGaussian Naive Bayes Results")
    print("-" * 35)
    for k, v in results.items():
        print(f"{k}: {v:.4f}" if k != "ML Model Name" else f"{k}: {v}")


if __name__ == "__main__":
    main()
