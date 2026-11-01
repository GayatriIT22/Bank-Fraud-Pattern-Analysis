# Bank Transaction Fraud Pattern Analysis

Big Data Analytics project (CSN443). Analysis of the PaySim synthetic mobile
money dataset to detect fraud patterns using Hadoop HDFS, Spark (PySpark) with
Hive-style SQL, and Spark MLlib Logistic Regression.

## Objectives

- Detect unusually high-value transactions
- Detect repeated failed attempts
- Detect location anomalies
- Detect suspicious activity (account draining, large night-time transfers)
- Compare rule-based detection with a Logistic Regression model

## Tools and Environment

- Hadoop 3.3.6 (single-node, pseudo-distributed HDFS) on Google Colab
- Apache Spark 4.0.4 (PySpark) with Hive-compatible SQL
- Java 17
- Python, pandas, matplotlib

## Dataset

PaySim synthetic mobile money dataset (Kaggle), about 6.36 million transactions
with 8,213 fraud cases (0.13%). Columns used: step, type, amount, nameOrig,
oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest,
isFraud.

The raw CSV (493 MB) is not included in this repository because of its size.

Assumptions: the PaySim dataset has no location or transaction status fields,
so `city` and `status` were added as synthetic columns, generated using the fraud
label to simulate attacker behavior. Pattern p2 and p3 are therefore affected by
this design and are not used as features in the ML model.

## Pipeline

1. Load the CSV into HDFS (`/fraud/raw/paysim/`)
2. Read with Spark and clean (drop nulls and duplicates, extract hour of day)
3. Create a Hive-style table and run SQL aggregations
4. Add synthetic `city` and `status` columns
5. Compute four rule-based patterns:
   - p1: amount above the 99th percentile for its transaction type
   - p2: city-hour burst of failed attempts (top 5%)
   - p3: transaction city differs from the account's home city
   - p4: large account drain, or large transfers between 1 AM and 4 AM
6. Evaluate each pattern against the fraud label (TP, FP, FN, precision, recall, F1)
7. Train Logistic Regression on a sample (all fraud plus 10% of normal rows)
8. Compare the model with rule p4 on the same test set

## Results

Pattern evaluation (full dataset):

| Pattern | Recall | Precision |
|---|---|---|
| p4 suspicious | 77.4% | 0.68% |
| any flag (any pattern) | 86.4% | 0.28% |

Rule vs model (test set, normal rows sampled at 10%):

| Method | TP | FP | FN | Precision (sample) | Recall |
|---|---|---|---|---|---|
| Rule p4 | 1308 | 18907 | 361 | 6.5% | 78.4% |
| Logistic Regression | 1490 | 4467 | 179 | 25.0% | 89.3% |

The Logistic Regression model has PR-AUC of 0.74. Precision on the sample is
inflated, because the normal class was downsampled. Full-data precision is
estimated at about 0.7% for the rule and about 3% for the model.

## Limitations

- Synthetic `city` and `status` columns are generated from the fraud label, so
  p2 and p3 results reflect the simulation design rather than real-world
  performance.
- Hour of day is `step % 24`, an approximation, because PaySim `step` is a
  simulation counter, not a clock time.
- Hadoop runs in single-node mode on Colab, not a multi-node cluster.
- Precision is high for recall-oriented rules only at the cost of many false
  positives. Threshold tuning is future work.

## Repository Structure

```
├── src/       Colab notebook (BigData_Fraud_Project.ipynb)
├── reports/   Graphs (pattern_eval.png, fraud_by_hour.png) and comparison CSV
├── sql/       Hive-style SQL queries
├── screenshots/ Report screenshots
├── data/      Raw CSV (not tracked, see .gitignore)
└── .gitignore
```

## How to Run

1. Open `src/BigData_Fraud_Project.ipynb` in Google Colab.
2. Upload the PaySim CSV to the Colab Files panel (not included in this repo).
3. Run the cells in order: HDFS setup, CSV upload to HDFS, Spark, Hive table,
   pattern analysis, Logistic Regression, and graphs.
