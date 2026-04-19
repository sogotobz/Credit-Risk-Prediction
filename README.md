# Credit-Risk-Prediction
Logistic regression model using the UCI Bank Marketing dataset to predict whether a client will subscribe to a term deposit (loan acceptance), with exploratory analysis of age, job, and marital status relationships to the target.

# Bank Marketing Loan Acceptance Prediction

This project uses the UCI Bank Marketing dataset to predict whether a client will accept a loan offer (`y`) using a logistic regression model. It includes exploratory data analysis (EDA), visualizations, feature encoding, model training, and evaluation.

## Dataset

The dataset is loaded from:

- `bank.csv` (semicolon-separated)

Key columns include:

- `age` — Client age  
- `job` — Job type  
- `marital` — Marital status  
- Other demographic and campaign-related features  
- `y` — Target variable (loan acceptance: yes/no)

Column names are cleaned to lowercase for consistency.

## Exploratory Data Analysis

The script explores:

- **Age distribution**  
  - Histogram of `age`
- **Age vs loan acceptance**  
  - Boxplot of `age` by `y`
- **Loan acceptance by job**  
  - Grouped proportions and stacked bar chart
- **Job distribution**  
  - Bar chart of job counts
- **Marital status distribution and acceptance**  
  - Bar charts and stacked bar chart by `marital` and `y`

These plots help understand how demographic factors relate to loan acceptance.

## Modeling

- Features: all columns except `y`
- Target: `y`
- Categorical variables are encoded using one-hot encoding (`pd.get_dummies`, `drop_first=True`).
- Train–test split: 80% train, 20% test (`random_state=42`)
- Model: `LogisticRegression(max_iter=1000)`

The model is trained on the encoded features and used to predict loan acceptance on the test set.

## Evaluation

The script prints:

- Model coefficients for each feature  
- Confusion matrix  
- Accuracy  
- Precision  
- Recall  
- Classification report

These metrics provide insight into how well the model distinguishes between clients who accept and do not accept the loan.

Author
Toba Sogo
