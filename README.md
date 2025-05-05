# 🏠 Real Estate Market Modeling 

The housing market significantly influences economic decision-making for individuals, investors, and policymakers. In this analysis, we examine an extensive real estate dataset (8GB+) supplemented with real-time data from Redfin to uncover patterns and forecast market trends. Leveraging Apache Spark and scalable machine learning pipelines, we apply regression models to predict housing prices, classify buyer behavior, and generate actionable insights. Our findings aim to inform smarter investment strategies, policy decisions, and home-buying choices in an increasingly dynamic market.
---

## 📁 Dataset

- **File**: `county_market_tracker.csv`
- **Source**:https://www.kaggle.com/datasets/thuynyle/redfin-housing-market-data​
- **Size**: ~8.5 GB 
- **Columns Include**:
  - `median_list_price`
  - `median_ppsf`
  - `inventory`
  - `homes_sold`
  - `pending_sales`
  - `new_listings`
  - `months_of_supply`
  - `median_dom`
  - `avg_sale_to_list`
  - `sold_above_list`
  - `price_drops`
  - `off_market_in_two_weeks`
  - `median_sale_price` 

---

## ⚙️ Tools & Technologies

- **Platform**: Databricks Community/Enterprise
- **Language**: Python (PySpark)
- **Libraries**:
  - `pyspark.ml`: LinearRegression, VectorAssembler, CrossValidator, TrainValidationSplit
  - `matplotlib` / `pandas` for visualization

---

## 📊 Workflow

### 1. **Data Loading & Cleaning**
- Read CSV from Databricks FileStore
- Clean and cast numeric fields
- Drop rows with nulls in target and feature columns

### 2. **Feature Engineering**
- Selected 12 features based on domain knowledge
- Assembled into a feature vector

### 3. **Model Training**
- Used **Linear Regression** model
- Data split: 80% training, 20% testing

### 4. **Hyperparameter Tuning**
- `regParam`: [0.01, 0.1]
- `elasticNetParam`: [0.0, 0.5, 1.0]
- Compared:
  - `CrossValidator` (3-fold)
  - `TrainValidationSplit` (80/20 split)

### 5. **Evaluation Metrics**
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)
- Coefficients and intercept used for feature importance analysis

---

## 🔍 Results


## 📈 Visualization

- Feature Importance (via coefficients)
- Optional scatter plots for individual feature vs. target (e.g., list price vs. sale price)







