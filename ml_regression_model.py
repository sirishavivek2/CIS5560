# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
import matplotlib.pyplot as plt

# Initialize Spark Session
# %pyspark
spark = SparkSession.builder \
    .appName("ML Regression Models") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
# File location and type
file_location = "/user/smahesh4/Sirisha-Project/merged_market_tracker.csv"
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
  .load(file_location)

df.show(10)

# COMMAND ----------

# Data Cleaning 

from pyspark.sql.functions import regexp_replace

numeric_cols = [
    "HOMES_SOLD", "NEW_LISTINGS", "INVENTORY", "MONTHS_OF_SUPPLY",
    "MEDIAN_LIST_PRICE", "MEDIAN_PPSF", "PENDING_SALES", "MEDIAN_SALE_PRICE"
]

for c in numeric_cols:
    df = df.withColumn(c, regexp_replace(col(c), ",", ""))
    df = df.withColumn(c, regexp_replace(col(c), "\\$", ""))
    df = df.withColumn(c, col(c).cast("double"))

df = df.dropna(subset=numeric_cols)



# COMMAND ----------

### Feature Engineering

#We use VectorAssembler to combine input columns into a single feature vector. We then scale the features using MinMaxScaler.
from pyspark.ml.feature import VectorAssembler, MinMaxScaler

feature_cols = [
    "HOMES_SOLD", "NEW_LISTINGS", "INVENTORY", 
    "MONTHS_OF_SUPPLY", "MEDIAN_LIST_PRICE", 
    "MEDIAN_PPSF", "PENDING_SALES"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = MinMaxScaler(inputCol="features", outputCol="normFeatures")

final_data = df.select(*feature_cols, col("MEDIAN_SALE_PRICE").alias("label")).dropna()



# COMMAND ----------

#Split Data
splits = final_data.randomSplit([0.8, 0.2], seed=42)
train_data = splits[0]
test_data = splits[1]


# COMMAND ----------

### NLP Pipeline (Tokenization and Stop Word Removal)

#This step is to tokenize text data and remove stop words using PySpark. We use the `RegexTokenizer` to split the text into alphabetic tokens of minimum 3 characters. Then we remove common stop words using `StopWordsRemover`. This prepares the text column for NLP analysis.

from pyspark.sql.functions import concat_ws

# Combine REGION, CITY, STATE, and PROPERTY_TYPE into a single text column
df = df.withColumn("DESCRIPTION", concat_ws(" ", "REGION", "CITY", "STATE", "PROPERTY_TYPE"))



# COMMAND ----------

# Tokenize Text Using RegexTokenizer
from pyspark.ml.feature import RegexTokenizer

# Tokenize using regex: alphabetic sequences, min length 3
tokenizer = RegexTokenizer() \
    .setPattern("\\p{L}+") \
    .setMinTokenLength(3) \
    .setGaps(False) \
    .setInputCol("DESCRIPTION") \
    .setOutputCol("words")

tokenized_df = tokenizer.transform(df)

# Show sample tokens
tokenized_df.select("DESCRIPTION", "words").show(5, truncate=False)



# COMMAND ----------

# Remove Stop Words
from pyspark.ml.feature import StopWordsRemover

# Use built-in English stop words
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
filtered_df = remover.transform(tokenized_df)

# Show cleaned tokens
filtered_df.select("words", "filtered").show(5, truncate=False)



# COMMAND ----------

# Topic Modeling with LDA
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.clustering import LDA

# Convert words to vectors
vectorizer = CountVectorizer(inputCol="filtered", outputCol="features")
vectorized_df = vectorizer.fit(filtered_df).transform(filtered_df)

# Fit LDA model
lda = LDA(k=5, maxIter=10)
lda_model = lda.fit(vectorized_df)

# Describe topics
topics = lda_model.describeTopics()
vocab = vectorizer.fit(filtered_df).vocabulary

topics.show()

# Print top words per topic
for row in topics.collect():
    print(f"Topic {row['topic']}: {[vocab[i] for i in row['termIndices']]}")




# COMMAND ----------

df.createOrReplaceTempView("market_data")



# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC
# MAGIC SELECT 
# MAGIC   region,
# MAGIC   COUNT(*) AS records,
# MAGIC   AVG(MEDIAN_SALE_PRICE) AS avg_price,
# MAGIC   MAX(MEDIAN_SALE_PRICE) AS max_price,
# MAGIC   MIN(MEDIAN_SALE_PRICE) AS min_price,
# MAGIC   STDDEV(MEDIAN_SALE_PRICE) AS price_stddev
# MAGIC FROM market_data
# MAGIC WHERE MEDIAN_SALE_PRICE IS NOT NULL
# MAGIC GROUP BY region
# MAGIC ORDER BY avg_price DESC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   YEAR(PERIOD_BEGIN) AS year,
# MAGIC   region,
# MAGIC   AVG(MEDIAN_SALE_PRICE) AS avg_price
# MAGIC FROM market_data
# MAGIC WHERE MEDIAN_SALE_PRICE IS NOT NULL
# MAGIC GROUP BY year, region
# MAGIC ORDER BY year, avg_price DESC
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   region,
# MAGIC   AVG(MEDIAN_SALE_PRICE) AS avg_price,
# MAGIC   RANK() OVER (ORDER BY AVG(MEDIAN_SALE_PRICE) DESC) AS price_rank
# MAGIC FROM market_data
# MAGIC WHERE MEDIAN_SALE_PRICE IS NOT NULL
# MAGIC GROUP BY region
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   region,
# MAGIC   AVG(MEDIAN_SALE_PRICE) AS avg_price,
# MAGIC   AVG(INVENTORY) AS avg_inventory
# MAGIC FROM market_data
# MAGIC WHERE MEDIAN_SALE_PRICE IS NOT NULL AND INVENTORY IS NOT NULL
# MAGIC GROUP BY region
# MAGIC ORDER BY avg_price DESC
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW clean_region_prices AS
# MAGIC SELECT region, AVG(MEDIAN_SALE_PRICE) AS avg_price
# MAGIC FROM market_data
# MAGIC WHERE MEDIAN_SALE_PRICE IS NOT NULL
# MAGIC GROUP BY region;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM clean_region_prices ORDER BY avg_price DESC;
# MAGIC

# COMMAND ----------

pandas_df = df.select("INVENTORY", "MEDIAN_SALE_PRICE") \
              .dropna() \
              .sample(False, 0.1) \
              .toPandas()


# COMMAND ----------



# COMMAND ----------

from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import TrainValidationSplit, CrossValidator, ParamGridBuilder
import time


# COMMAND ----------

dt = DecisionTreeRegressor(labelCol="label", featuresCol="normFeatures")
pipeline = Pipeline(stages=[assembler, scaler, dt])


# COMMAND ----------

paramGrid = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [5, 10]) \
    .addGrid(dt.maxBins, [32, 64]) \
    .build()


# COMMAND ----------

start_time_tvs = time.time()

tvs = TrainValidationSplit(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse"),
    trainRatio=0.8
)

model_tvs = tvs.fit(train_data)
end_time_tvs = time.time()


# COMMAND ----------

start_time_cv = time.time()

cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse"),
    numFolds=3
)

model_cv = cv.fit(train_data)
end_time_cv = time.time()


# COMMAND ----------

pred_tvs = model_tvs.transform(test_data)
pred_cv = model_cv.transform(test_data)

evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

print("----- TVS Model Evaluation -----")
print("RMSE:", evaluator_rmse.evaluate(pred_tvs))
print("R²:", evaluator_r2.evaluate(pred_tvs))
print("Execution Time (TVS): {:.2f} seconds".format(end_time_tvs - start_time_tvs))

print("\n----- CV Model Evaluation -----")
print("RMSE:", evaluator_rmse.evaluate(pred_cv))
print("R²:", evaluator_r2.evaluate(pred_cv))
print("Execution Time (CV): {:.2f} seconds".format(end_time_cv - start_time_cv))


# COMMAND ----------

best_model = model_tvs.bestModel.stages[-1]  # Get the Decision Tree stage
feature_importances = best_model.featureImportances

# Print feature importance mapped to column names
print("\nFeature Importances:")
for i, col_name in enumerate(feature_cols):
    print(f"{col_name}: {feature_importances[i]:.4f}")


# COMMAND ----------

# Select prediction and actual label from TVS model
prediction_vs_actual = pred_tvs.select("prediction", "label")

# Show sample rows
prediction_vs_actual.show(10)
# Convert to pandas for visualization
pdf = prediction_vs_actual.toPandas()



# COMMAND ----------

from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import TrainValidationSplit, CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import time


# COMMAND ----------

gbt = GBTRegressor(labelCol="label", featuresCol="normFeatures", maxIter=50)
pipeline_gbt = Pipeline(stages=[assembler, scaler, gbt])


# COMMAND ----------

paramGrid_gbt = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [3, 5]) \
    .addGrid(gbt.maxIter, [20, 50]) \
    .build()


# COMMAND ----------

start_time_tvs_gbt = time.time()

tvs_gbt = TrainValidationSplit(
    estimator=pipeline_gbt,
    estimatorParamMaps=paramGrid_gbt,
    evaluator=RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse"),
    trainRatio=0.8
)

model_tvs_gbt = tvs_gbt.fit(train_data)
end_time_tvs_gbt = time.time()


# COMMAND ----------

start_time_cv_gbt = time.time()

cv_gbt = CrossValidator(
    estimator=pipeline_gbt,
    estimatorParamMaps=paramGrid_gbt,
    evaluator=RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse"),
    numFolds=3
)

model_cv_gbt = cv_gbt.fit(train_data)
end_time_cv_gbt = time.time()


# COMMAND ----------

pred_tvs_gbt = model_tvs_gbt.transform(test_data)
pred_cv_gbt = model_cv_gbt.transform(test_data)

print("----- GBT TVS Model Evaluation -----")
print("RMSE:", evaluator_rmse.evaluate(pred_tvs_gbt))
print("R²:", evaluator_r2.evaluate(pred_tvs_gbt))
print("Execution Time (TVS): {:.2f} seconds".format(end_time_tvs_gbt - start_time_tvs_gbt))

print("\n----- GBT CV Model Evaluation -----")
print("RMSE:", evaluator_rmse.evaluate(pred_cv_gbt))
print("R²:", evaluator_r2.evaluate(pred_cv_gbt))
print("Execution Time (CV): {:.2f} seconds".format(end_time_cv_gbt - start_time_cv_gbt))


# COMMAND ----------

best_model_gbt = model_tvs_gbt.bestModel.stages[-1]  # GBT stage
gbt_importances = best_model_gbt.featureImportances.toArray()

print("\nGBT Feature Importances:")
for i, col_name in enumerate(feature_cols):
    print(f"{col_name}: {gbt_importances[i]:.4f}")







#
