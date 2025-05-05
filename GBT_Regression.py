from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType
from pyspark.sql.functions import *

from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler

# Start Spark Session
spark = SparkSession.builder.appName("GradientBoostingModel").getOrCreate()

# File location and type
file_location = "/user/sruiz85/Group5-Project/county_market_tracker.csv"
file_location = "/user/sruiz85/Group5-Project/state_market_tracker.csv"
file_location = "/user/sruiz85/Group5-Project/us_national_market_tracker.csv"

file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# Read CSVs
county_df = spark.read.csv("/user/sruiz85/Group5-Project/county_market_tracker.csv", header=True, inferSchema=True)
state_df = spark.read.csv("/user/sruiz85/Group5-Project/state_market_tracker.csv", header=True, inferSchema=True)
us_df = spark.read.csv("/user/sruiz85/Group5-Project/us_national_market_tracker.csv", header=True, inferSchema=True)

df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

df.show()

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# Create a view or table

temp_table_name = "sale_price_csv"

df.createOrReplaceTempView(temp_table_name)

data = spark.sql("SELECT * FROM sale_price_csv")

# Check schema, nulls, summary stats:
df.printSchema()
df.describe().show()
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()

# Select features (Independent Variables)
feature_cols = ['homes_sold', 'pending_sales', 'new_listings', 'inventory', 'months_of_supply', 'median_list_price', 'median_ppsf', 'median_list_ppsf', 'price_drops', 'avg_sale_to_list', 'sold_above_list', 'median_dom', 'off_market_in_two_weeks', 'inventory_yoy', 'median_sale_price_yoy']  # to capture trend
target_col = 'median_sale_price'

# Filter dataset to only needed columns
model_df = df.select(feature_cols + [target_col])

# Median sale price over time
df.select("period_begin", "median_sale_price") \
  .orderBy("period_begin") \
  .show(10, truncate=False)

# Distribution of "median_sale_price"
df.select("median_sale_price").summary().show()

# Distribution of "homes_sold"
df.select("homes_sold").summary().show()

# Distribution of "inventory"
df.select("inventory").summary().show()

# Distribution of "months_of_supply"
df.select("months_of_supply").summary().show()

# Distribution of "price_drops"
df.select("price_drops").summary().show()

"""### 3. Correlation Heatmap"""

numeric_cols = [
    "median_sale_price", "homes_sold", "pending_sales", "new_listings", "inventory",
    "months_of_supply", "median_list_price", "median_ppsf", "median_list_ppsf",
    "price_drops", "avg_sale_to_list", "sold_above_list", "median_dom",
    "off_market_in_two_weeks", "inventory_yoy", "median_sale_price_yoy"
]

"""### Data Cleaning"""

# Cleans the data (drops nulls)
model_df = model_df.dropna()

"""### Assemble Features into Vector"""

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
assembled_df = assembler.transform(model_df).select("features", target_col)

"""### Split the Data
It is common practice when building supervised machine learning models to split the source data, using some of it to train the model and reserving some to test the trained model. In this project, we will use 80% of the data for training, and reserve 20% for testing.
"""

splits = assembled_df.randomSplit([0.8, 0.2], seed=42)
train_df = splits[0]
test_df = splits[1]

# Count the rows
train_rows = train_df.count()
test_rows = test_df.count()

print("Training Rows:", train_rows, " Testing Rows:", test_rows)

"""### Train Regression Model
To train the regression model, you need a training data set that includes a vector of numeric features, and a label column. Here, we will use the **VectorAssembler** class to transform the feature columns into a vector, and then rename the **median_sales_price** column to **label**.
"""

gb = GBTRegressor(featuresCol="features", labelCol=target_col)
gb_model = gb.fit(train_df)

"""### Evaluate the Model
Now you're ready to use the **transform** method of the model to generate some predictions. You can use this approach to predict median sales price where the label is unknown; but in this case you are using the test data which includes a known true label value, so you can compare the predicted median sales price.
"""

predictions = gb_model.transform(test_df)

evaluator_rmse = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")

"""Model Performance:
RMSE ≈ 8,488 → On average, our predictions are off by about $8,488.

R² ≈ 0.977 → Our model explains about 97.7% of the variance in median_sale_price.

🔍 What This Means:
Very high R²: Strong fit — the model is capturing nearly all the signal.

Low RMSE: Given typical U.S. housing prices, an ~8K error is small (often <3% of home value), indicating strong predictive power.

### Feature Importance
"""

# Let us see which features GBT used most
for feature, importance in zip(feature_cols, gb_model.featureImportances.toArray()):
    print(f"{feature}: {importance:.4f}")

"""####🔝 Most Influential Feature
- median_list_price: 0.7626

Hugely dominant — the model relies on this one feature more than all others combined.

This makes sense: listed prices often set buyer expectations and anchor market behavior.

####🧠 Moderately Important Features
These contribute in meaningful, but smaller ways:

- median_sale_price_yoy: 0.0294

- inventory_yoy: 0.0274

- off_market_in_two_weeks: 0.0272

- months_of_supply: 0.0267

These all reflect market dynamics and pricing momentum — strong supporting signals for the model.

#### 👥 Less Important, But Still Useful
median_ppsf, median_list_ppsf, price_drops, sold_above_list, median_dom, etc. — in the 0.01–0.02 range

These probably add slight refinements or local variability

####💤 Least Useful in This Model
pending_sales: 0.0053

homes_sold: 0.0090

Surprisingly low importance — either:

They’re correlated with other stronger features

Or they don't add much signal beyond what’s already captured

## Let us do a train a model without median_list_price to see how dependent our results are on it.
"""

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# 1. New feature list (excluding median_list_price)
reduced_feature_cols = [
    "homes_sold", "pending_sales", "new_listings", "inventory", "months_of_supply",
    "median_ppsf", "median_list_ppsf", "price_drops", "avg_sale_to_list",
    "sold_above_list", "median_dom", "off_market_in_two_weeks",
    "inventory_yoy", "median_sale_price_yoy"
]

# 2. Rebuild dataset
reduced_df = df.select(reduced_feature_cols + [target_col]).dropna()

# 3. Assemble new features
reduced_assembler = VectorAssembler(inputCols=reduced_feature_cols, outputCol="features")
reduced_assembled = reduced_assembler.transform(reduced_df).select("features", target_col)

# 4. Split into train/test
train_df2, test_df2 = reduced_assembled.randomSplit([0.8, 0.2], seed=42)

# 5. Train GBT without median_list_price
gb2 = GBTRegressor(featuresCol="features", labelCol=target_col, maxIter=100)
gb_model2 = gb2.fit(train_df2)

# 6. Evaluate performance
predictions2 = gb_model2.transform(test_df2)

rmse2 = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="rmse").evaluate(predictions2)
r2_2 = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="r2").evaluate(predictions2)

print(f"Without median_list_price -> RMSE: {rmse2}")
print(f"Without median_list_price -> R²: {r2_2}")

"""Performance Comparison
Model	                       RMSE	       R²
With median_list_price	     8,488	     0.977
Without median_list_price	   15,054	     0.928

####🔍 Interpretation
✅ Still a good model without median_list_price (R² ≈ 93% is solid)

⚠️ But you lose ~44% accuracy in RMSE terms — that's a huge jump in error

🔥 median_list_price is clearly the primary driver of sale prices in your dataset

####🧠 Takeaway
Your model works with or without median_list_price, but:

It relies heavily on it for peak accuracy

Other features contribute, but none match its predictive power

### Cross-Validation with GBTRegressor
"""

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor

# Reuse the feature-assembled dataset (e.g., assembled_df or reduced_assembled)
# We'll use the full dataset (no manual split — CV handles it)
cv_dataset = assembled_df  # Or use reduced_assembled if excluding median_list_price

# 1. Define model
gbt = GBTRegressor(featuresCol="features", labelCol=target_col)

# 2. Define evaluator
evaluator = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="rmse")

# 3. Create parameter grid
param_grid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [3, 5]) \
    .addGrid(gbt.maxIter, [50]) \
    .build()

# 4. Define CrossValidator
cv = CrossValidator(
    estimator=gbt,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3,  # 3-fold cross-validation
    seed=42
)

# 5. Run cross-validation
cv_model = cv.fit(cv_dataset)

# 6. Evaluate best model
best_model = cv_model.bestModel
predictions = best_model.transform(cv_dataset)

rmse_cv = evaluator.evaluate(predictions)
r2_cv = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="r2").evaluate(predictions)

print(f"Cross-Validated RMSE: {rmse_cv}")
print(f"Cross-Validated R²: {r2_cv}")

"""### TrainValidationSplit with GBTRegressor"""

from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor

# 1. Define model
gbt = GBTRegressor(featuresCol="features", labelCol=target_col)

# 2. Define evaluator
evaluator = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="rmse")

# 3. Define parameter grid
param_grid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [3, 5]) \
    .addGrid(gbt.maxIter, [50]) \
    .build()

# 4. Define TrainValidationSplit
tvs = TrainValidationSplit(
    estimator=gbt,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    trainRatio=0.8,  # 80% for training, 20% for validation
    seed=42
)

# 5. Fit on full dataset
tvs_model = tvs.fit(assembled_df)

# 6. Evaluate best model
best_model_tvs = tvs_model.bestModel
predictions_tvs = best_model_tvs.transform(assembled_df)

rmse_tvs = evaluator.evaluate(predictions_tvs)
r2_tvs = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="r2").evaluate(predictions_tvs)

print(f"TrainValidationSplit RMSE: {rmse_tvs}")
print(f"TrainValidationSplit R²: {r2_tvs}")

"""#### Visual comparison across all three Gradient Boosting runs"""

import pandas as pd

# Create Pandas DataFrame for visualization
comparison_df = pd.DataFrame({
    "Model": ["GBTRegressor", "GBTRegressor (CrossValidator)", "GBTRegressor (TrainValidationSplit)"],
    "RMSE": [rmse, rmse_cv, rmse_tvs],
    "R2": [r2, r2_cv, r2_tvs]
})

# Convert to Spark DataFrame for display
comparison_sdf = spark.createDataFrame(comparison_df)
comparison_sdf.show(10, truncate=False)

"""#### 🔍 Key Observations:
✅ RMSE dropped by ~77% after tuning.

✅ R² rose to near-perfect accuracy.

⚖️ CrossValidator and TrainValidationSplit yielded identical best models, meaning the same hyperparameter combination was optimal in both cases.

#### Add Computation Time & Hyperparameters
"""

import time

dataset = assembled_df

# -- Base GBTRegressor
start = time.time()
gb_model = GBTRegressor(featuresCol="features", labelCol=target_col).fit(dataset)
base_duration = time.time() - start

# -- CrossValidator
start = time.time()
cv_model = cv.fit(dataset)
cv_duration = time.time() - start

# -- TrainValidationSplit
start = time.time()
tvs_model = tvs.fit(dataset)
tvs_duration = time.time() - start

import pandas as pd

# Add best parameters
cv_best = cv_model.bestModel
tvs_best = tvs_model.bestModel

cv_maxDepth = cv_best.getOrDefault("maxDepth")
cv_maxIter = cv_best.getOrDefault("maxIter")
tvs_maxDepth = tvs_best.getOrDefault("maxDepth")
tvs_maxIter = tvs_best.getOrDefault("maxIter")

comparison_df = pd.DataFrame({
    "Model": ["GBTRegressor", "GBTRegressor (CrossValidator)", "GBTRegressor (TrainValidationSplit)"],
    "RMSE": [rmse, rmse_cv, rmse_tvs],
    "R2": [r2, r2_cv, r2_tvs],
    "Time (s)": [base_duration, cv_duration, tvs_duration],
    "Best maxDepth": [None, cv_maxDepth, tvs_maxDepth],
    "Best maxIter": [None, cv_maxIter, tvs_maxIter]
})

# Convert to Spark DataFrame for Databricks visualization
comparison_sdf = spark.createDataFrame(comparison_df)
df.show(comparison_sdf)
