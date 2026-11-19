# Machine Learning Classification Model Comparison

## 1. Problem Statement

The objective of this project is to implement multiple machine learning classification models using one classification dataset and compare their performance using standard evaluation metrics.

The project implements the following five classification models:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier
5. Random Forest Classifier (Ensemble)

Each model is evaluated using:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## 2. Dataset Description

### Dataset Name

**UCI Adult / Census Income Dataset**

### Dataset Source

UCI Machine Learning Repository

### Dataset Link

https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data

### Problem Type

Binary classification.

### Target Variable

The target variable is:

`income`

The target is converted into binary values for model training:

- `<=50K` → `0`
- `>50K` → `1`

### Why This Dataset Was Selected

The dataset contains more than the minimum required number of instances and features and contains both numerical and categorical variables. It also provides missing values and class-distribution characteristics that allow the required preprocessing and class-imbalance analysis to be demonstrated.

### Feature Types

#### Numerical Features

- age
- fnlwgt
- education_num
- capital_gain
- capital_loss
- hours_per_week

#### Categorical Features

- workclass
- education
- marital_status
- occupation
- relationship
- race
- sex
- native_country

### Dataset Analysis

The project examines:

- Number of instances
- Number of features
- Feature types
- Missing values
- Duplicate records
- Class distribution

Actual dataset statistics should be updated here after execution.

---

## 3. GitHub Repository Link

**Repository:** [ADD YOUR GITHUB REPOSITORY LINK HERE]

Example:

`https://github.com/<username>/ML_Assignment_2_Classification`

---

## 4. Data Preprocessing

The following preprocessing steps are applied before model training.

### 4.1 Missing Values

Missing values are identified and handled.

- Numerical features: median imputation
- Categorical features: most-frequent imputation

### 4.2 Duplicate Records

Duplicate records are identified and removed.

### 4.3 Categorical Encoding

Categorical features are converted into numerical representation using OneHotEncoder.

`handle_unknown="ignore"` is used so that unseen categories in the test data do not cause an error.

### 4.4 Feature Scaling

Numerical features are standardized using StandardScaler where required.

### 4.5 Train-Test Split

The dataset is divided into:

- 80% training data
- 20% testing data

The split is stratified to maintain the class distribution.

### 4.6 Fixed Random Seed

A fixed random seed is used for reproducibility.

`random_state = 42`

### 4.7 Data Leakage Prevention

Preprocessing is fitted only on the training data and then applied to the test data.

---

## 5. Models Used

### 5.1 Logistic Regression

Logistic Regression is used as a baseline binary classification model.

Source file:

`model/logistic_regression.py`

### 5.2 Decision Tree Classifier

Decision Tree Classifier predicts the target class using a sequence of feature-based decision rules.

Source file:

`model/decision_tree.py`

### 5.3 K-Nearest Neighbor Classifier

KNN predicts the class of an observation based on the classes of its nearest neighboring observations.

Source file:

`model/knn.py`

### 5.4 Naive Bayes Classifier

Gaussian Naive Bayes is used as the Naive Bayes implementation.

Source file:

`model/naive_bayes.py`

### 5.5 Random Forest Classifier

Random Forest is the ensemble model used in this project and combines multiple decision trees.

Source file:

`model/random_forest.py`

---

## 6. Evaluation Metrics

Each model is evaluated using the following metrics.

### Accuracy

Measures the proportion of correctly classified observations.

### AUC Score

Measures the model's ability to distinguish between the classes.

### Precision

Measures how many observations predicted as positive are actually positive.

### Recall

Measures how many actual positive observations are correctly identified.

### F1 Score

Provides the harmonic mean of Precision and Recall.

### Matthews Correlation Coefficient (MCC)

Provides a balanced measure of classification quality using all four confusion-matrix categories.

---

## 7. Model Comparison

The actual results below must be replaced with the values obtained from the executed models.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | [RESULT] | [RESULT] | [RESULT] | [RESULT] | [RESULT] | [RESULT] |
| Decision Tree | [RESULT] | [RESULT] | [RESULT] | [RESULT] | [RESULT] | [RESULT] |
| KNN | [RESULT] | [RESULT] | [RESULT] | [RESULT] | [RESULT] | [RESULT] |
| Naive Bayes | [RESULT] | [RESULT] | [RESULT] | [RESULT] | [RESULT] | [RESULT] |
| Random Forest (Ensemble) | [RESULT] | [RESULT] | [RESULT] | [RESULT] | [RESULT] | [RESULT] |

> Note: The assignment lists five required models. The comparison-table wording mentions six models, but only five model names are specified. This README therefore follows the five explicitly named required models rather than inventing a sixth model.

---

## 8. Visualization

The performance of the five classification models is represented graphically.

The project includes:

1. Model performance comparison chart
2. Metric comparison visualization / heatmap

The visualization compares:

- Accuracy
- AUC
- Precision
- Recall
- F1
- MCC

Add the final generated graph images here after running the project.

Example:

`results/model_comparison.png`

---

## 9. Observations

### Logistic Regression

**Observation:** [ADD OBSERVATION FROM ACTUAL RESULTS]

### Decision Tree

**Observation:** [ADD OBSERVATION FROM ACTUAL RESULTS]

### KNN

**Observation:** [ADD OBSERVATION FROM ACTUAL RESULTS]

### Naive Bayes

**Observation:** [ADD OBSERVATION FROM ACTUAL RESULTS]

### Random Forest

**Observation:** [ADD OBSERVATION FROM ACTUAL RESULTS]

---

## 10. Overall Winner

### Overall Winner

**[ADD MODEL NAME AFTER ACTUAL EXECUTION]**

### Reason

The overall winner should be selected based on the actual comparison of Accuracy, AUC, Precision, Recall, F1 Score and MCC.

---

## 11. Project Structure

```text
ML_Assignment_2_Classification/
│
├── README.md
├── requirements.txt
├── test_data.csv
│
└── model/
    ├── ml_pipeline_utils.py
    ├── logistic_regression.py
    ├── decision_tree.py
    ├── knn.py
    ├── naive_bayes.py
    └── random_forest.py
```

---

## 12. Requirements

The project uses the following Python libraries:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
```

Install the dependencies using:

```bash
pip install -r requirements.txt
```

---

## 13. Execution Instructions

### Google Colab

1. Open the project notebook in Google Colab.
2. Install/import the required libraries.
3. Load the selected dataset.
4. Perform dataset analysis.
5. Perform preprocessing.
6. Perform the 80:20 stratified train-test split.
7. Train all five models.
8. Calculate the six required metrics.
9. Generate the comparison table and visualizations.
10. Export the test data used in the experiment.

### Python / Local Environment

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project:

```bash
cd ML_Assignment_2_Classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the individual models:

```bash
python model/logistic_regression.py
python model/decision_tree.py
python model/knn.py
python model/naive_bayes.py
python model/random_forest.py
```

---

## 14. Test Data

The test data used in the experiments is stored in:

`test_data.csv`

This file should contain the test data used for the final model evaluation.

---

## 15. Reproducibility

The project uses a fixed random seed:

```text
random_state = 42
```

The same dataset and consistent preprocessing approach are used for all five classification models.

---

## 16. BITS Virtual Lab Execution

The final project must also be executed in the BITS Virtual Lab.

The required execution screenshot should be captured from the BITS Virtual Lab and included in the final submission PDF.

---

## 17. Conclusion

This project implements and compares five machine learning classification algorithms using the same dataset and consistent preprocessing and evaluation methods.

The comparison is based on Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient.

The final conclusion and overall winning model will be determined from the actual experimental results.

---

## 18. Academic Integrity

This project should be reviewed, understood, tested, and adapted by the student before submission.

All reported metrics, observations, visualizations, and conclusions must come from the student's own execution of the project.
