# Car Price Prediction using Machine Learning

> A college-level Data Science mini project that builds a **Linear Regression model** to predict car prices based on technical and physical specifications.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Results & Evaluation](#results--evaluation)
- [Key Insights](#key-insights)
- [Best Practices Applied](#best-practices-applied)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Project Overview

The automobile industry is highly competitive, and pricing a car correctly is a critical business decision. This project uses **supervised machine learning** to predict the selling price of a car based on its features.

| Property | Details |
|---|---|
| **Problem Type** | Supervised Learning — Regression |
| **Algorithm** | Linear Regression (Ordinary Least Squares) |
| **Target Variable** | `price` (in USD) |
| **Dataset Size** | 205 rows × 26 columns |

---

## Dataset

**File:** `CarPrice_Assignment.csv`

The dataset contains information about various car models including their physical dimensions, engine specifications, fuel type, and market price.

### Numerical Columns (16)

| Column | Description |
|---|---|
| `symboling` | Insurance risk rating (-3 to +3) |
| `wheelbase` | Distance between front and rear axles (inches) |
| `carlength` | Length of the car (inches) |
| `carwidth` | Width of the car (inches) |
| `carheight` | Height of the car (inches) |
| `curbweight` | Weight of the car without occupants (lbs) |
| `enginesize` | Size of the engine (cc) |
| `boreratio` | Bore-to-stroke ratio |
| `stroke` | Volume of the engine stroke |
| `compressionratio` | Engine compression ratio |
| `horsepower` | Engine horsepower |
| `peakrpm` | Peak revolutions per minute |
| `citympg` | Miles per gallon in city driving |
| `highwaympg` | Miles per gallon on highway |
| `price` |  **Target variable** — car price in USD |

### Categorical Columns (10)

| Column | Description |
|---|---|
| `CarName` | Car manufacturer and model name |
| `fueltype` | Gas or diesel |
| `aspiration` | Standard or turbo |
| `doornumber` | Two or four doors |
| `carbody` | Convertible, sedan, hatchback, wagon, hardtop |
| `drivewheel` | Front / rear / four-wheel drive |
| `enginelocation` | Front or rear engine |
| `enginetype` | Type of engine (dohc, ohc, etc.) |
| `cylindernumber` | Number of cylinders |
| `fuelsystem` | Fuel injection system type |

### Dataset Notes

-  No missing values
-  No duplicate rows
-  Feature engineering applied: car brand extracted from `CarName`
-  Brand name typos corrected (e.g. `maxda` → `mazda`, `vw` → `volkswagen`)

---

##  Project Structure

```
car-price-prediction/
│
├── CarPrice_Assignment.csv              # Raw dataset
├── Car_Price_Prediction_Project.ipynb  # Main Jupyter / Colab notebook
└── README.md                            # This file
```

---

##  Notebook Walkthrough

The notebook follows a structured, step-by-step format aligned with the academic evaluation criteria:

| # | Section | Description | Marks |
|---|---|---|---|
| 1 | Problem Understanding | Business context, objective, ML problem type | 2 |
| 2 | Import Libraries | pandas, numpy, matplotlib, seaborn, sklearn | — |
| 3 | Load Dataset | Load CSV, confirm shape | — |
| 4 | Initial Exploration | `head()`, `info()`, `describe()` | 2 |
| 5 | Feature Understanding | Numerical vs categorical split | — |
| 6 | Data Cleaning | Missing values, duplicates, drop irrelevant columns | 3 |
| 7 | Feature Engineering | Extract & clean brand name from `CarName` | — |
| 8 | EDA | Heatmap, brand prices, body style, scatter plots | 3 |
| 9 | Encoding | Label Encoding for all categorical features | — |
| 10 | Feature Selection | Separate X (features) and y (target) | — |
| 11 | Train-Test Split | 80% train / 20% test, `random_state=42` | — |
| 12 | Feature Scaling | StandardScaler (fit on train only) | — |
| 13 | Model Selection | Linear Regression explanation | 4 |
| 14 | Train Model | `model.fit()` on scaled training data | — |
| 15 | Predictions | `model.predict()` on test set | — |
| 16 | Model Evaluation | R², MAE, MSE, RMSE + relative error | 3 |
| 17 | Visualization | Actual vs Predicted, Residuals, Feature Importance | 2 |
| 18 | Conclusion | Key findings, best practices, future scope | 1 |

**Total: 20 Marks**

---

##  Technologies Used

| Library | Version | Purpose |
|---|---|---|
| `Python` | 3.8+ | Core programming language |
| `pandas` | latest | Data loading and manipulation |
| `numpy` | latest | Numerical computations |
| `matplotlib` | latest | Base plotting library |
| `seaborn` | latest | Statistical visualizations |
| `scikit-learn` | latest | ML model, preprocessing, evaluation |

---

##  How to Run

### Option 1 — Google Colab (Recommended)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload Notebook** and select `Car_Price_Prediction_Project.ipynb`
3. Upload `CarPrice_Assignment.csv` using the Files panel on the left sidebar
4. Update the dataset path in **Section 3** if needed:
   ```python
   df = pd.read_csv('CarPrice_Assignment.csv')
   ```
5. Click **Runtime → Run All**

### Option 2 — Local Jupyter Notebook

**Step 1 — Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

**Step 2 — Launch Jupyter**
```bash
jupyter notebook
```

**Step 3 — Open the notebook**

Navigate to `Car_Price_Prediction_Project.ipynb` in the Jupyter interface and run all cells.

---

##  Results & Evaluation

The model is evaluated using four standard regression metrics:

| Metric | Formula | Interpretation |
|---|---|---|
| **R² Score** | 1 − (SS_res / SS_tot) | % of price variance explained by the model |
| **MAE** | mean(\|actual − predicted\|) | Average error in USD (easy to interpret) |
| **MSE** | mean((actual − predicted)²) | Penalises large errors more than MAE |
| **RMSE** | √MSE | Error in original USD units |

> Run the notebook to see the exact metric values for your train-test split.

### Visualizations Generated

-  Price Distribution (histogram + boxplot)
-  Correlation Heatmap (numerical features)
-  Average Price by Brand (bar chart)
-  Price by Body Style and Fuel Type (boxplots)
-  Engine Size vs Price colored by Horsepower (scatter)
-  Actual vs Predicted Prices (with ideal line)
-  Residuals vs Predicted + Residual Distribution
-  Top 15 Features by Regression Coefficient

---

##  Key Insights

1. **Engine size, curb weight, and horsepower** are the strongest positive predictors of car price — bigger, more powerful cars cost more.

2. **Fuel efficiency (citympg, highwaympg)** has a negative correlation with price — economy-focused cars tend to be cheaper.

3. **Brand matters significantly** — luxury brands like Jaguar, Buick, and Porsche command premium prices irrespective of specs.

4. **Body style influences pricing** — hardtop and convertible styles are priced higher than hatchbacks and wagons on average.

5. **Linear Regression** is a suitable baseline because several key features exhibit strong linear relationships with price.

---

##  Best Practices Applied

| Practice | Details |
|---|---|
| No data leakage | StandardScaler fitted only on training data |
| Reproducibility | `random_state=42` set for train-test split |
| Feature engineering | Brand extracted and typos corrected |
| Code clarity | Every code cell has inline comments |
| Residual analysis | Model assumptions validated visually |
| Multiple metrics | R², MAE, MSE, and RMSE all reported |
| Relative error | RMSE compared to mean price for context |

---

##  Future Improvements

- Apply **log transformation** to `price` to reduce right skewness
- Use **One-Hot Encoding** instead of Label Encoding for nominal features
- Try **Ridge / Lasso Regression** to handle potential multicollinearity
- Explore **non-linear models** — Random Forest, XGBoost — for better accuracy
- Implement **k-Fold Cross Validation** for a more robust evaluation
- Add **hyperparameter tuning** with GridSearchCV

---

##  Author

**Project Type:** College Data Science Mini Project  
**Algorithm:** Linear Regression  
**Subject:** Machine Learning / Data Science  

---

*"All models are wrong, but some are useful." — George Box*

