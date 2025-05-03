from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, TrainValidationSplit, ParamGridBuilder

import pandas as pd

# Start Spark session
spark = SparkSession.builder.appName("LinearRegressionModel").getOrCreate()

# Set file path
file_type = "csv"
file_path = "/user/smahesh4/Group5-Project/county_market_tracker.csv"

# Load DataFrame
df = spark.read.format(file_type).option("header", "true").option("inferSchema", "true").load(file_path)

# Show data
df.show()

# Define label and features
label_col = "median_sale_price"
feature_cols = [
    "median_list_price", "median_ppsf", "inventory", "homes_sold", "pending_sales",
    "new_listings", "months_of_supply", "median_dom", "avg_sale_to_list",
    "sold_above_list", "price_drops", "off_market_in_two_weeks"
]

# Clean invalid or non-numeric values before casting
for c in feature_cols + [label_col]:
    df = df.withColumn(
        c,
        when(col(c).rlike("^[0-9.]+$"), col(c)).otherwise(None).cast("double")
    )

# Drop rows with nulls in feature or label columns
df = df.na.drop(subset=feature_cols + [label_col])
print(f"Row count after dropping nulls: {df.count()}")

# Assemble features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(df).select("features", label_col)

# Train/test split
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Fit Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol=label_col)
lr_model = lr.fit(train_data)

# Parameter grid for tuning
param_grid = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.01, 0.1])\
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
    .build()

evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")

# CrossValidator
cv = CrossValidator(estimator=lr,
                    estimatorParamMaps=param_grid,
                    evaluator=evaluator,
                    numFolds=3,
                    parallelism=2)
cv_model = cv.fit(train_data)
cv_predictions = cv_model.transform(test_data)
cv_rmse = evaluator.evaluate(cv_predictions)
cv_r2 = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2").evaluate(cv_predictions)

# TrainValidationSplit
tvs = TrainValidationSplit(estimator=lr,
                           estimatorParamMaps=param_grid,
                           evaluator=evaluator,
                           trainRatio=0.8)
tvs_model = tvs.fit(train_data)
tvs_predictions = tvs_model.transform(test_data)
tvs_rmse = evaluator.evaluate(tvs_predictions)
tvs_r2 = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2").evaluate(tvs_predictions)

# Final model predictions
predictions = lr_model.transform(test_data)

evaluator_rmse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

print("=== Model Comparison ===")
print(f"CrossValidator:     RMSE = {cv_rmse:.2f}, R² = {cv_r2:.4f}")
print(f"TrainValidationSplit: RMSE = {tvs_rmse:.2f}, R² = {tvs_r2:.4f}")

print("Intercept:", lr_model.intercept)
print("Coefficients:", lr_model.coefficients)

# Coefficient importance
features_list = feature_cols
coefficients = lr_model.coefficients.toArray().tolist()

importance_df = pd.DataFrame({
    "Feature": features_list,
    "Coefficient": coefficients,
    "Importance (abs)": [abs(c) for c in coefficients]
}).sort_values(by="Importance (abs)", ascending=False)

print("\n=== Feature Importances ===")
print(importance_df.to_string(index=False))

from tabulate import tabulate

# Evaluation Metrics Table
metrics_data = [
    ["LinearRegression", rmse, r2],
    ["CrossValidator", cv_rmse, cv_r2],
    ["TrainValidationSplit", tvs_rmse, tvs_r2]
]

print("\n=== Evaluation Metrics ===")
print(tabulate(metrics_data, headers=["Model", "RMSE", "R²"], tablefmt="grid", floatfmt=".4f"))

# Feature Importances Table
importance_data = [
    [f, c, abs(c)] for f, c in zip(feature_cols, coefficients)
]
importance_data.sort(key=lambda x: x[2], reverse=True)

print("\n=== Feature Importances ===")
print(tabulate(importance_data, headers=["Feature", "Coefficient", "Importance (abs)"], tablefmt="grid", floatfmt=".6f"))

