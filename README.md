# Car Price Prediction using Machine Learning

A college-level Data Science mini project that builds a **Linear
Regression model** to predict car prices based on technical and physical
specifications.

------------------------------------------------------------------------

# Table of Contents

-   Project Overview
-   Dataset
-   Dataset Loading Strategy
-   Project Structure
-   Notebook Walkthrough
-   Technologies Used
-   How to Run
-   Results & Evaluation
-   Key Insights
-   Best Practices Applied
-   Future Improvements
-   Author

------------------------------------------------------------------------

# Project Overview

The automobile industry is highly competitive, and pricing a car
correctly is an important business decision. This project applies
supervised machine learning techniques to estimate the market price of a
car based on its technical specifications.

  Property          Details
  ----------------- --------------------------------------------
  Problem Type      Supervised Learning --- Regression
  Algorithm         Linear Regression (Ordinary Least Squares)
  Target Variable   price
  Dataset Size      205 rows × 26 columns

The model learns relationships between vehicle specifications such as
engine size, horsepower, fuel type, and physical dimensions to estimate
the expected price of a car.

------------------------------------------------------------------------

# Dataset

File: `CarPrice_Assignment.csv`

The dataset contains information about various car models including
their physical dimensions, engine specifications, fuel type, and market
price.

## Numerical Columns

  Column             Description
  ------------------ ---------------------------------------
  symboling          Insurance risk rating
  wheelbase          Distance between front and rear axles
  carlength          Length of the car
  carwidth           Width of the car
  carheight          Height of the car
  curbweight         Weight of the car without occupants
  enginesize         Size of the engine
  boreratio          Bore-to-stroke ratio
  stroke             Engine stroke volume
  compressionratio   Engine compression ratio
  horsepower         Engine horsepower
  peakrpm            Peak revolutions per minute
  citympg            Fuel efficiency in city driving
  highwaympg         Fuel efficiency on highways
  price              Target variable (car price)

## Categorical Columns

  Column           Description
  ---------------- ------------------------
  CarName          Manufacturer and model
  fueltype         Gas or diesel
  aspiration       Standard or turbo
  doornumber       Number of doors
  carbody          Body style
  drivewheel       Drive type
  enginelocation   Engine position
  enginetype       Engine configuration
  cylindernumber   Number of cylinders
  fuelsystem       Fuel injection system

------------------------------------------------------------------------

# Dataset Loading Strategy

The dataset can be loaded from two different sources to ensure
flexibility across environments such as Google Colab, Jupyter Notebook,
or local development setups.

``` python
try:
    # Try loading from uploaded/local path
    df = pd.read_csv('CarPrice_Assignment.csv')
    print("Dataset loaded from uploaded file")

except:
    # Fallback: load from GitHub URL
    url = "https://github.com/VishnurajVishwakarama/DATASETS/blob/main/CarPrice_Assignment.csv"
    df = pd.read_csv(url)
    print("Dataset loaded from GitHub URL")

print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
```

------------------------------------------------------------------------

# Project Structure

    car-price-prediction/
    │
    ├── CarPrice_Assignment.csv
    ├── Car_Price_Prediction_Project.ipynb
    └── README.md

------------------------------------------------------------------------

# Notebook Walkthrough

  -----------------------------------------------------------------------
  Step                    Section                 Description
  ----------------------- ----------------------- -----------------------
  1                       Problem Understanding   Define the business
                                                  objective

  2                       Import Libraries        Import Python libraries

  3                       Load Dataset            Load and verify dataset

  4                       Initial Exploration     Explore structure and
                                                  statistics

  5                       Feature Understanding   Identify numerical and
                                                  categorical features

  6                       Data Cleaning           Remove duplicates and
                                                  irrelevant fields

  7                       Feature Engineering     Extract car brand

  8                       Exploratory Data        Identify patterns and
                          Analysis                correlations

  9                       Encoding                Convert categorical
                                                  features

  10                      Feature Selection       Separate predictors and
                                                  target

  11                      Train-Test Split        Split dataset

  12                      Feature Scaling         Apply StandardScaler

  13                      Model Training          Fit Linear Regression

  14                      Predictions             Predict prices

  15                      Model Evaluation        Calculate metrics

  16                      Visualization           Plot predictions and
                                                  residuals

  17                      Conclusion              Summarize findings
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# Technologies Used

  Technology     Purpose
  -------------- ---------------------------
  Python         Programming language
  Pandas         Data manipulation
  NumPy          Numerical computation
  Matplotlib     Data visualization
  Seaborn        Statistical visualization
  Scikit-learn   Machine learning modeling

------------------------------------------------------------------------

# How to Run

## Google Colab

1.  Upload the notebook to Google Colab
2.  Upload the dataset file if required
3.  Run all cells

## Local Jupyter

Install dependencies:

    pip install pandas numpy matplotlib seaborn scikit-learn jupyter

Launch Jupyter:

    jupyter notebook

Open the notebook and run all cells.

------------------------------------------------------------------------

# Results & Evaluation

The model is evaluated using standard regression metrics:

-   R² Score
-   Mean Absolute Error (MAE)
-   Mean Squared Error (MSE)
-   Root Mean Squared Error (RMSE)

Visualizations include price distribution, correlation heatmaps,
brand-wise pricing, prediction comparison plots, and residual analysis.

------------------------------------------------------------------------

# Key Insights

1.  Engine size, curb weight, and horsepower are strong predictors of
    car price.
2.  Fuel efficiency tends to negatively correlate with price.
3.  Brand reputation significantly influences pricing.
4.  Certain body styles command higher market value.
5.  Linear Regression serves as a strong baseline model.

------------------------------------------------------------------------

# Best Practices Applied

-   Prevented data leakage by scaling only training data
-   Used reproducible train-test split
-   Applied feature engineering
-   Evaluated model using multiple metrics
-   Performed residual analysis

------------------------------------------------------------------------

# Future Improvements

-   Apply log transformation to the target variable
-   Use One-Hot Encoding
-   Try Ridge and Lasso Regression
-   Explore Random Forest and XGBoost
-   Apply cross-validation and hyperparameter tuning

------------------------------------------------------------------------

# Author

Vishnuraj Vishwakarma

Data Science Mini Project Machine Learning -- Linear Regression
