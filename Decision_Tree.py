from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, TrainValidationSplit, ParamGridBuilder
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder \
    .appName("RealEstateData") \
    .getOrCreate()

# Load dataset from HDFS
file_path = "/user/smahesh4/Group5-Project/RealEstateData.csv"
df = spark.read.format("csv") \
    .option("inferSchema", "true") \
    .option("header", "true") \
    .option("sep", ",") \
    .load(file_path)

print("=== Sample Data ===")
df.show(5, truncate=False)

# Drop unnecessary columns
df_cleaned = df.drop(
    'Unnamed: 58', 'PERIOD_BEGIN59', 'PERIOD_END60',
    'REGION_TYPE61', 'REGION_TYPE_ID62', 'LAST_UPDATED98'
)

# Define target and features
target_col = 'median_sale_price13'
feature_cols = ['inventory', 'new_listings', 'homes_sold', 'median_list_price']

# Cast and clean
df_model = df_cleaned.select([col(c).cast("double") for c in [target_col] + feature_cols])
df_model = df_model.na.drop()

# Assemble features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
assembled_data = assembler.transform(df_model).select("features", col(target_col).alias("label"))

# Split data
train_data, test_data = assembled_data.randomSplit([0.8, 0.2], seed=42)

# Model
dt = DecisionTreeRegressor(featuresCol="features", labelCol="label", maxDepth=5)
dt_model = dt.fit(train_data)

# Predict and evaluate
predictions = dt_model.transform(test_data)
evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

# Feature importance
importances = dt_model.featureImportances
features = assembler.getInputCols()
importance_df_pd = pd.DataFrame({
    "Feature": features,
    "Importance": importances.toArray()
}).sort_values(by="Importance", ascending=False)
importance_df = spark.createDataFrame(importance_df_pd)
print("=== Feature Importances ===")
importance_df.show(truncate=False)

# Train-validation split
val_data = assembled_data.subtract(train_data)
paramGrid = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [3, 5, 7]) \
    .addGrid(dt.minInstancesPerNode, [1, 2, 4]) \
    .build()

tvs = TrainValidationSplit(estimator=dt,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator_rmse,
                           trainRatio=0.8)
tvs_model = tvs.fit(train_data)
tvs_predictions = tvs_model.transform(val_data)
tvs_rmse = evaluator_rmse.evaluate(tvs_predictions)
print("=== Train Validation ===")
importance_df.show(truncate=False)

# Cross-validation
cv = CrossValidator(estimator=dt,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator_rmse,
                    numFolds=5)
cv_model = cv.fit(train_data)
cv_predictions = cv_model.transform(val_data)
cv_rmse = evaluator_rmse.evaluate(cv_predictions)
print("=== Cross Validation ===")
importance_df.show(truncate=False)
# Output summary as table
summary_rows = [
    ("Modeling", "DecisionTreeRegressor with depth and param tuning","In-Memory"),
    ("Training", "Model trained on 80% of data","train_data"),
    ("Testing", "Predictions on test set","test_data"),
    ("Evaluation (RMSE)", "Root Mean Squared Error",f"{rmse:.2f}"),
    ("Evaluation (MAE)", "Mean Absolute Error",f"{mae:.2f}"),
    ("Evaluation (R²)", "R-squared Score",f"{r2:.2f}"),
    ("Feature Importance", "Extracted and printed","/output/feature_importance"),
    ("TrainValidationSplit", "Validation RMSE from TVS", f"{tvs_rmse:.2f}"),
    ("CrossValidation", "Cross-validated RMSE (5 folds)", f"{cv_rmse:.2f}"),
   
]


# Convert summary to DataFrame and show
summary_df = spark.createDataFrame(summary_rows, ["Component", "Description", "Output"])
print("\n=== Pipeline Summary ===")
summary_df.show(truncate=False)

# New input for future prediction
new_data = spark.createDataFrame([
    (1200.0, 300.0, 280.0, 450000.0),
    (950.0, 250.0, 260.0, 430000.0)
], ['inventory', 'new_listings', 'homes_sold', 'median_list_price'])

# Assemble features
new_assembled = assembler.transform(new_data).select("features")

# Predict
future_predictions = dt_model.transform(new_assembled)

# Show predictions
print("\n=== Future Predictions ===")
future_predictions.select("prediction").show(truncate=False)


spark.stop()
