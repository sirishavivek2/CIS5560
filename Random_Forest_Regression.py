# Random Forest Regression Pipeline with Logging and Evaluation Tables
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, Imputer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
import pandas as pd
from tabulate import tabulate
from datetime import timedelta
import time

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Neighborhood Market Tracker") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Load Data
file_path = "/user/smahesh4/Group5-Project/neighborhood_market_tracker.csv"
df = spark.read.option("header", True).option("inferSchema", True).csv(file_path)

# Select Features and Label
feature_cols = ["inventory", "homes_sold", "median_list_price", "median_ppsf"]
label_col = "median_sale_price"

# Cast and Clean Data
for c in feature_cols + [label_col]:
    df = df.withColumn(c, col(c).cast("double"))

df = df.na.drop(subset=feature_cols + [label_col])

# Impute Missing Values
imputer = Imputer(inputCols=feature_cols, outputCols=feature_cols)
df = imputer.fit(df).transform(df)

# Assemble Features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(df).select("features", label_col)

# Split Data
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Define Evaluators
rmse_evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
r2_evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2")

# === Baseline Random Forest (no tuning) ===
baseline_rf = RandomForestRegressor(labelCol=label_col, featuresCol="features")
baseline_start = time.time()
baseline_model = baseline_rf.fit(train_data)
baseline_end = time.time()
baseline_time = baseline_end - baseline_start
baseline_predictions = baseline_model.transform(test_data)
baseline_rmse = rmse_evaluator.evaluate(baseline_predictions)
baseline_r2 = r2_evaluator.evaluate(baseline_predictions)

# Define Model and Param Grid
rf = RandomForestRegressor(labelCol=label_col, featuresCol="features")
param_grid = ParamGridBuilder().addGrid(rf.numTrees, [5]).addGrid(rf.maxDepth, [3]).build()

# === CrossValidator ===
cv = CrossValidator(estimator=rf, estimatorParamMaps=param_grid, evaluator=rmse_evaluator, numFolds=2, parallelism=1)
cv_start = time.time()
cv_model = cv.fit(train_data)
cv_end = time.time()
cv_time = cv_end - cv_start
cv_predictions = cv_model.transform(test_data)
cv_rmse = rmse_evaluator.evaluate(cv_predictions)
cv_r2 = r2_evaluator.evaluate(cv_predictions)

# === TrainValidationSplit ===
tvs = TrainValidationSplit(estimator=rf, estimatorParamMaps=param_grid, evaluator=rmse_evaluator, trainRatio=0.8)
tvs_start = time.time()
tvs_model = tvs.fit(train_data)
tvs_end = time.time()
tvs_time = tvs_end - tvs_start
tvs_predictions = tvs_model.transform(test_data)
tvs_rmse = rmse_evaluator.evaluate(tvs_predictions)
tvs_r2 = r2_evaluator.evaluate(tvs_predictions)

# === Feature Importances ===
best_model = cv_model.bestModel
importances = best_model.featureImportances.toArray()
importance_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values(by="Importance", ascending=False)
print("\n=== Feature Importances ===")
print(tabulate(importance_df.values.tolist(), headers=["Feature", "Importance"], tablefmt="grid", floatfmt=".6f"))

# Format time
def format_time(seconds):
    return str(timedelta(seconds=round(seconds)))

# === Evaluation Metrics Table ===
metrics_data = [
    ["RandomForest (Baseline)", baseline_rmse, baseline_r2, baseline_time, format_time(baseline_time)],
    ["CrossValidator", cv_rmse, cv_r2, cv_time, format_time(cv_time)],
    ["TrainValidationSplit", tvs_rmse, tvs_r2, tvs_time, format_time(tvs_time)]
]
print("\n=== Evaluation Metrics ===")
print(tabulate(metrics_data, headers=["Model", "RMSE", "R²", "Training Time (s)", "Formatted Time"], tablefmt="grid", floatfmt=".4f"))

# === Best Hyperparameters ===
print("\n=== Best Hyperparameters from CrossValidator ===")
if hasattr(best_model, "getNumTrees"):
    print(f"Best numTrees: {best_model.getNumTrees}")
if best_model.getOrDefault("maxDepth") is not None:
    print(f"Best maxDepth: {best_model.getOrDefault('maxDepth')}")

# === Completeness Table ===
completeness_data = [
    ["RandomForest (Baseline)", "Completed", f"{baseline_rmse:.4f}", f"{baseline_r2:.4f}", format_time(baseline_time)],
    ["CrossValidator", "Completed", f"{cv_rmse:.4f}", f"{cv_r2:.4f}", format_time(cv_time)],
    ["TrainValidationSplit", "Completed", f"{tvs_rmse:.4f}", f"{tvs_r2:.4f}", format_time(tvs_time)]
]
print("\n=== Completeness of Modeling, Training, Testing, Evaluation ===")
print(tabulate(completeness_data, headers=["Stage", "Status", "RMSE", "R²", "Training Time"], tablefmt="grid"))
