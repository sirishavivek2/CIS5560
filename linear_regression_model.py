# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, TrainValidationSplit, ParamGridBuilder
import pandas as pd
import time
from tabulate import tabulate
from datetime import timedelta


# Start Spark session
spark = SparkSession.builder.appName("LinearRegressionModel").getOrCreate()

# Set file path
file_type = "csv"
file_path = "/user/smahesh4/Group5-Project/county_market_tracker.csv"

# Load data
df = spark.read.format(file_type).option("header", "true").option("inferSchema", "true").load(file_path)
df.show()

# Define label and features
label_col = "median_sale_price"
feature_cols = [
    "median_list_price", "median_ppsf", "inventory", "homes_sold", "pending_sales",
    "new_listings", "months_of_supply", "median_dom", "avg_sale_to_list",
    "sold_above_list", "price_drops", "off_market_in_two_weeks"
]

# Clean and cast values
for c in feature_cols + [label_col]:
    df = df.withColumn(
        c, when(col(c).rlike("^[0-9.]+$"), col(c)).otherwise(None).cast("double")
    )

df = df.na.drop(subset=feature_cols + [label_col])
print(f"Row count after dropping nulls: {df.count()}")

# Assemble features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(df).select("features", label_col)

# Split data
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Define Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol=label_col)

# Fit baseline model
lr_start = time.time()
lr_model = lr.fit(train_data)
lr_end = time.time()

# Predict and evaluate
predictions = lr_model.transform(test_data)
evaluator_rmse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2")
rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)
lr_time = lr_end - lr_start

# Param grid
param_grid = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.01, 0.1])\
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
    .build()

# === CrossValidator ===
cv = CrossValidator(estimator=lr,
                    estimatorParamMaps=param_grid,
                    evaluator=evaluator_rmse,
                    numFolds=3,
                    parallelism=2)

cv_start = time.time()
cv_model = cv.fit(train_data)
cv_end = time.time()
cv_time = cv_end - cv_start

cv_predictions = cv_model.transform(test_data)
cv_rmse = evaluator_rmse.evaluate(cv_predictions)
cv_r2 = evaluator_r2.evaluate(cv_predictions)

# === TrainValidationSplit ===
tvs = TrainValidationSplit(estimator=lr,
                           estimatorParamMaps=param_grid,
                           evaluator=evaluator_rmse,
                           trainRatio=0.8)

tvs_start = time.time()
tvs_model = tvs.fit(train_data)
tvs_end = time.time()
tvs_time = tvs_end - tvs_start

tvs_predictions = tvs_model.transform(test_data)
tvs_rmse = evaluator_rmse.evaluate(tvs_predictions)
tvs_r2 = evaluator_r2.evaluate(tvs_predictions)

# Print model results
print("\n=== Linear Regression Coefficients ===")
print("Intercept:", lr_model.intercept)
print("Coefficients:", lr_model.coefficients)

# === Feature Importances ===
features_list = feature_cols
coefficients = lr_model.coefficients.toArray().tolist()
importance_df = pd.DataFrame({
    "Feature": features_list,
    "Coefficient": coefficients,
    "Importance (abs)": [abs(c) for c in coefficients]
}).sort_values(by="Importance (abs)", ascending=False)

print("\n=== Feature Importances ===")
print(tabulate(importance_df.values.tolist(),
               headers=["Feature", "Coefficient", "Importance (abs)"],
               tablefmt="grid", floatfmt=".6f"))


# === Evaluation Metrics Table ===
def format_time(seconds):
    return str(timedelta(seconds=round(seconds)))

metrics_data = [
    ["LinearRegression", rmse, r2, lr_time, format_time(lr_time)],
    ["CrossValidator", cv_rmse, cv_r2, cv_time, format_time(cv_time)],
    ["TrainValidationSplit", tvs_rmse, tvs_r2, tvs_time, format_time(tvs_time)],
]

print("\n=== Evaluation Metrics ===")
print(tabulate(metrics_data,
               headers=["Model", "RMSE", "R²", "Training Time (s)", "Formatted Time"],
               tablefmt="grid", floatfmt=".4f"))

print("\n=== Completeness of Modeling Workflow ===")
print("Modeling (Linear Regression) Completed")
print(f"   - RMSE: {rmse:.4f}")
print(f"   - R²: {r2:.4f}")
print(f"   - Training Time: {format_time(lr_time)}")

print("\n Cross Validation Completed")
print(f"   - Best RMSE: {cv_rmse:.4f}")
print(f"   - Best R²: {cv_r2:.4f}")
print(f"   - Training Time: {format_time(cv_time)}")

print("\n TrainValidationSplit Completed")
print(f"   - Best RMSE: {tvs_rmse:.4f}")
print(f"   - Best R²: {tvs_r2:.4f}")
print(f"   - Training Time: {format_time(tvs_time)}")

print("\nAll stages (Modeling, Training, Testing, Evaluation) have been successfully executed.")

# Determine best model by lowest RMSE
best_model_name, best_rmse, best_r2 = min(
    [("LinearRegression", rmse, r2), ("CrossValidator", cv_rmse, cv_r2), ("TrainValidationSplit", tvs_rmse, tvs_r2)],
    key=lambda x: x[1]  # minimize RMSE
)



# Table-style completeness summary
completeness_data = [
    ["LinearRegression", "Completed", f"{rmse:.4f}", f"{r2:.4f}", format_time(lr_time)],
    ["CrossValidator", "Completed", f"{cv_rmse:.4f}", f"{cv_r2:.4f}", format_time(cv_time)],
    ["TrainValidationSplit", "Completed", f"{tvs_rmse:.4f}", f"{tvs_r2:.4f}", format_time(tvs_time)]
]

print("\n=== Completeness of Modeling, Training, Testing, Evaluation ===")
print(tabulate(completeness_data,
               headers=["Stage", "Status", "RMSE", "R²", "Training Time"],
               tablefmt="grid"))




