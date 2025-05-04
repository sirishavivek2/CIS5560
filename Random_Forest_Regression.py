# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import Imputer
from pyspark.sql.functions import col,when
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import time
import pandas as pd
import matplotlib.pyplot as plt

# Start Spark session
spark = SparkSession.builder \
    .appName("Neighborhood Market Tracker") \
    .getOrCreate()

# Load data
# File location and type
file_path = "/user/smahesh4/Group5-Project/neighborhood_market_tracker.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_path)
df.show(5)
df.printSchema()


#  Define features and label BEFORE using them
feature_cols = ["inventory", "homes_sold", "median_list_price", "median_ppsf"]
label_col = "median_sale_price"
# Cast to Double
for c in feature_cols + [label_col]:
    df = df.withColumn(c, when(col(c).rlike("^[0-9.]+$"), col(c)).otherwise(None).cast("double"))

# Drop null label rows
df = df.na.drop(subset=[label_col])

# Impute missing feature values
imputer = Imputer(inputCols=feature_cols, outputCols=feature_cols)
df = imputer.fit(df).transform(df)


# COMMAND ----------

#Feature Importance 
# Assemble features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(df).select("features", label_col)


# COMMAND ----------

#Train-Test Split
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)


# COMMAND ----------

#Model Definition
rf = RandomForestRegressor(labelCol=label_col, featuresCol="features")


# COMMAND ----------

#Hyperparameter Tuning Grids
param_grid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50]) \
    .addGrid(rf.maxDepth, [5]) \
    .build()



# COMMAND ----------

#. Cross Validation (CV)
from pyspark.ml.evaluation import RegressionEvaluator
# Define the evaluator
evaluator = RegressionEvaluator(
    labelCol=label_col,
    predictionCol="prediction",
    metricName="rmse"
)
cv = CrossValidator(
    estimator=rf,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3,
    parallelism=2
)

start_time = time.time()
cv_model = cv.fit(train_data)
end_time = time.time()

predictions = cv_model.transform(test_data)


# COMMAND ----------

#TrainValidationSplit (TVS)
from pyspark.ml.tuning import TrainValidationSplit
# Define TrainValidationSplit
tvs = TrainValidationSplit(
    estimator=rf,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    trainRatio=0.8
)



tvs = TrainValidationSplit(
    estimator=rf,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    trainRatio=0.8
)
# Fit and predict
tvs_model = tvs.fit(train_data)
tvs_predictions = tvs_model.transform(test_data)


# COMMAND ----------

#Model Evaluation
r2_evaluator = RegressionEvaluator(
    labelCol=label_col,
    predictionCol="prediction",
    metricName="r2"
)
# RMSE and R2 for CV
rmse = evaluator.evaluate(predictions)
r2 = r2_evaluator.evaluate(predictions)

# RMSE and R2 for TVS
tvs_rmse = evaluator.evaluate(tvs_predictions)
tvs_r2 = r2_evaluator.evaluate(tvs_predictions)


# COMMAND ----------

# Fit CrossValidator
cv_model = cv.fit(train_data)

# ✅ Must assign this
best_model = cv_model.bestModel

# Now you can safely use:
print("Best numTrees:", best_model.getNumTrees)
print("Best maxDepth:", best_model.getOrDefault("maxDepth"))

# COMMAND ----------

#Feature Importance 
importances = best_model.featureImportances.toArray()

feature_importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\n=== Feature Importances ===")
print(feature_importance_df.to_string(index=False))
spark_feature_importance_df = spark.createDataFrame(feature_importance_df)
spark_feature_importance_df.show(truncate=False)




# COMMAND ----------

#Output Summary in Table Format
from pyspark.sql import Row

summary_rows = [
    Row(Stage="Train/Test", Metric="RMSE", Value=float(rmse)),
    Row(Stage="Train/Test", Metric="R2", Value=float(r2)),
    Row(Stage="CV", Metric="Training Time (s)", Value=float(end_time - start_time)),
    Row(Stage="CV", Metric="Best NumTrees", Value=float(best_model.getNumTrees)),
    Row(Stage="CV", Metric="Best MaxDepth", Value=float(best_model.getOrDefault("maxDepth"))),
    Row(Stage="TVS", Metric="RMSE", Value=float(tvs_rmse)),
    Row(Stage="TVS", Metric="R2", Value=float(tvs_r2))
]
spark.createDataFrame(feature_importance_df).show(truncate=False)
summary_df = spark.createDataFrame(summary_rows)
summary_df.show(truncate=False)


