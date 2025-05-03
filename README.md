# 🏠 Real Estate Market Modeling with PySpark

This project uses a dataset of U.S. county-level real estate market indicators to predict **median sale prices** using **linear regression** and **hyperparameter tuning techniques**. The project is developed and run on **Databricks**, leveraging **Apache Spark** for scalable data processing and machine learning.

---

## 📁 Dataset

- **File**: `county_market_tracker.csv`
- **Source**: Uploaded to Databricks FileStore (`/FileStore/tables/`)
- **Size**: ~XX MB (dependent on actual file)
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
  - `median_sale_price` *(target variable)*

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

| Model                         | RMSE     | R²       |
|------------------------------|----------|----------|
| LinearRegression (CV)        | ~X.XX    | ~0.XX    |
| LinearRegression (TVS)       | ~X.XX    | ~0.XX    |

- **Top Influential Features**:
  - `median_list_price`
  - `median_ppsf`
  - `avg_sale_to_list`
  - `sold_above_list`
  - `months_of_supply` (negative influence)

---

## 📈 Visualization

- Feature Importance (via coefficients)
- Optional scatter plots for individual feature vs. target (e.g., list price vs. sale price)

---

## 🚀 How to Run

1. Upload `county_market_tracker.csv` to `/FileStore/tables/` in Databricks
2. Import the notebook and attach a cluster
3. Run all cells in order
4. View model performance and insights

---

## 📌 License

This project is intended for educational and research purposes. Please attribute data sources appropriately if reused.

---

## ✍️ Author

- Your Name (or GitHub handle)
- [LinkedIn/GitHub/Portfolio link]

