# Decision Tree Regression Pipeline with Structured Output
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, TrainValidationSplit, ParamGridBuilder
import pandas as pd
from datetime import timedelta
import time
from tabulate import tabulate

# Initialize Spark session
spark = SparkSession.builder \
    .appName("DecisionTreeModel") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Load dataset
file_path = "/user/smahesh4/Group5-Project/RealEstateData.csv"
df = spark.read.option("header", True).option("inferSchema", True).csv(file_path)

# Clean and prepare data
df_cleaned = df.drop(
    'Unnamed: 58', 'PERIOD_BEGIN59', 'PERIOD_END60',
    'REGION_TYPE61', 'REGION_TYPE_ID62', 'LAST_UPDATED98'
)

target_col = 'median_sale_price13'
feature_cols = ['inventory', 'new_listings', 'homes_sold', 'median_list_price']

# Cast columns and drop nulls
df_model = df_cleaned.select([col(c).cast("double") for c in [target_col] + feature_cols]).na.drop()

# Assemble features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(df_model).select(col("features"), col(target_col).alias("label"))

# Split data
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Baseline Decision Tree Model
dt = DecisionTreeRegressor(featuresCol="features", labelCol="label", maxDepth=5)
dt_start = time.time()
dt_model = dt.fit(train_data)
dt_end = time.time()
dt_time = dt_end - dt_start

# Predictions & Evaluations
predictions = dt_model.transform(test_data)
evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

# Feature Importance
importances = dt_model.featureImportances.toArray()
importance_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values(by="Importance", ascending=False)

# Hyperparameter Tuning
param_grid = ParamGridBuilder().addGrid(dt.maxDepth, [3, 5, 7]).build()

# TrainValidationSplit
tvs = TrainValidationSplit(estimator=dt, estimatorParamMaps=param_grid, evaluator=evaluator_rmse, trainRatio=0.8)
tvs_start = time.time()
tvs_model = tvs.fit(train_data)
tvs_end = time.time()
tvs_time = tvs_end - tvs_start
tvs_predictions = tvs_model.transform(test_data)
tvs_rmse = evaluator_rmse.evaluate(tvs_predictions)
tvs_r2 = evaluator_r2.evaluate(tvs_predictions)

# CrossValidator
cv = CrossValidator(estimator=dt, estimatorParamMaps=param_grid, evaluator=evaluator_rmse, numFolds=3)
cv_start = time.time()
cv_model = cv.fit(train_data)
cv_end = time.time()
cv_time = cv_end - cv_start
cv_predictions = cv_model.transform(test_data)
cv_rmse = evaluator_rmse.evaluate(cv_predictions)
cv_r2 = evaluator_r2.evaluate(cv_predictions)

# Format time helper
def format_time(seconds):
    return str(timedelta(seconds=round(seconds)))


# === Evaluation Metrics ===
print("\n=== Evaluation Metrics ===")
eval_time_data = [
    ["Model", "RMSE", "R²", "Time (s)", "Formatted Time"],
    ["DecisionTree", f"{rmse:.2f}", f"{r2:.2f}", f"{dt_time:.2f}", format_time(dt_time)],
    ["TrainValidationSplit", f"{tvs_rmse:.2f}", f"{tvs_r2:.2f}", f"{tvs_time:.2f}", format_time(tvs_time)],
    ["CrossValidator", f"{cv_rmse:.2f}", f"{cv_r2:.2f}", f"{cv_time:.2f}", format_time(cv_time)]
]
print(tabulate(eval_time_data, headers="firstrow", tablefmt="grid"))


# === Feature Importances ===
print("\n=== Feature Importances ===")
feature_table = [[f, f"{i:.6f}"] for f, i in zip(importance_df.Feature, importance_df.Importance)]
print(tabulate(["Feature", "Importance"], tablefmt="grid"))
print(tabulate(feature_table, tablefmt="grid"))

# === Computation Times ===
print("\n=== Computation Time Summary ===")
time_data = [
    ["Stage", "Time (s)", "Formatted Time", "RMSE", "R²"],
    ["DecisionTree", f"{dt_time:.2f}", format_time(dt_time), f"{rmse:.2f}", f"{r2:.2f}"],
    ["TrainValidationSplit", f"{tvs_time:.2f}", format_time(tvs_time), f"{tvs_rmse:.2f}", f"{tvs_r2:.2f}"],
    ["CrossValidator", f"{cv_time:.2f}", format_time(cv_time), f"{cv_rmse:.2f}", f"{cv_r2:.2f}"]
]
print(tabulate(time_data, headers="firstrow", tablefmt="grid"))

