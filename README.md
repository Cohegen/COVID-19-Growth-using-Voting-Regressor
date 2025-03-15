# COVID-19 Case Growth Prediction using Voting Regressor

## 📌 Project Overview
This project aims to predict the growth of COVID-19 cases using an ensemble learning approach, specifically the **Voting Regressor**. A Voting Regressor is a combination of multiple regression models that work together to improve predictive accuracy.

## 📊 Dataset Description
The dataset is sourced from the **Johns Hopkins University COVID-19 Dataset**. It contains time-series data of daily confirmed COVID-19 cases worldwide. Key features include:
- **Date**: The reporting date of COVID-19 cases.
- **Confirmed Cases**: The total number of confirmed cases at a given time.
- **Other potential features**: Deaths, recoveries, region-based cases, etc.

## 🛠 Methodology & Models Used
1. **Data Preprocessing**: Cleaning and transforming the dataset.
2. **Feature Engineering**: Selecting independent variables that influence COVID-19 case growth.
3. **Model Selection**: The Voting Regressor is built using:
   - Linear Regression
   - Random Forest Regression
   - Support Vector Regression (SVR)
4. **Model Training**: Training individual regressors and combining them.
5. **Evaluation**: Comparing the model's performance using Mean Squared Error (MSE) and R² Score.

## 🚀 Implementation Steps
Importing Libraries

Loading the Dataset

Preprocessing the data

Defining our dependent and independent variable

Feature Scaling

Training individual Regressors

Implementing voting Regressors

Compare model Performance

Plotting all models

## 🔧 How to Run the Code

1. Install the required dependencies:
```bash
pip install pandas numpy matplotlib scikit-learn
```

2. Load the dataset and preprocess it.
3. Train individual regression models.
4. Combine models using the Voting Regressor.
5. Evaluate and visualize predictions.

## 📈 Results & Analysis
The results of the Voting Regressor are compared against individual models to assess its effectiveness. Predictions are visualized to show the trend of COVID-19 case growth.

## 🏁 Conclusion
This project demonstrates how ensemble learning improves prediction accuracy for COVID-19 case growth. Using a Voting Regressor, we leverage the strengths of multiple regression models to achieve better results.

## 🛠 Code Overview

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
```

```python
# Load COVID-19 confirmed cases dataset
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
df = pd.read_csv(url)

# Display first few rows
df.head()
```

```python
# Select a country (change if needed)
country = "Kenya"
df_country = df[df["Country/Region"] == country].drop(columns=["Province/State", "Lat", "Long"]).sum()

# Convert to time series
df_country = df_country.iloc[1:]  # Remove 'Country/Region' row
df_country.index = pd.to_datetime(df_country.index)  # Convert dates
df_country = df_country.reset_index()
df_country.columns = ["Date", "Confirmed"]

# Add "Days Since First Case" column
df_country["Days"] = (df_country["Date"] - df_country["Date"].m
```

## 🤝 Contributing
Feel free to fork, modify, and improve the project!

## 📜 License
This project is open-source under the MIT License.
