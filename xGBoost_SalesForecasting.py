#%%
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from xgboost import plot_importance

# Import training and test data
train = pd.read_csv('../DemandForecasting/demand_forecasting_test_train/train.csv')
test = pd.read_csv('../DemandForecasting/demand_forecasting_test_train/test.csv')

# DATES FEATURES
def date_features(df):
    # Date Features
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df.date.dt.year
    df['month'] = df.date.dt.month
    df['day'] = df.date.dt.day
    df['dayofyear'] = df.date.dt.dayofyear
    df['dayofweek'] = df.date.dt.dayofweek
    df['weekofyear'] = df.date.dt.isocalendar().week
    
    # Additionnal Data Features
    df['day^year'] = np.log((np.log(df['dayofyear'] + 1)) ** (df['year'] - 2000))
    
    # Drop date
    df.drop('date', axis=1, inplace=True)
    
    return df

# Dates Features for Train, Test
train, test = date_features(train), date_features(test)

#%%
# Daily Average, Monthly Average for train
train['daily_avg']  = train.groupby(['item','store','dayofweek'])['sales'].transform('mean')
train['monthly_avg'] = train.groupby(['item','store','month'])['sales'].transform('mean')
train = train.dropna()

# Average sales for Day_of_week = d per Item,Store
daily_avg = train.groupby(['item','store','dayofweek'])['sales'].mean().reset_index()
# Average sales for Month = m per Item,Store
monthly_avg = train.groupby(['item','store','month'])['sales'].mean().reset_index()

#%%
# Merge Test with Daily Avg, Monthly Avg
def merge(df1, df2, col,col_name):
    
    df1 =pd.merge(df1, df2, how='left', on=None, left_on=col, right_on=col,
            left_index=False, right_index=False, sort=True,
             copy=True, indicator=False)
    
    df1 = df1.rename(columns={'sales':col_name})
    return df1

# Add Daily_avg and Monthly_avg features to test 
test = merge(test, daily_avg,['item','store','dayofweek'],'daily_avg')
test = merge(test, monthly_avg,['item','store','month'],'monthly_avg')

# Sales Rolling mean sequence per item 
rolling_10 = train.groupby(['item'])['sales'].rolling(10).mean().reset_index().drop('level_1', axis=1)
train['rolling_mean'] = rolling_10['sales'] 

# 90 last days of training rolling mean sequence added to test data
rolling_last90 = train.groupby(['item','store'])['rolling_mean'].tail(90).copy()
test['rolling_mean'] = rolling_last90.reset_index().drop('index', axis=1)

# Shifting rolling mean 3 months
train['rolling_mean'] = train.groupby(['item'])['rolling_mean'].shift(90) # Create a feature with rolling mean of day - 90
train.head()
# train.tail()

#%%
# Clean features highly correlated to each others
for df in [train, test]:
    df.drop(['dayofyear', 
                  'weekofyear',
                  'daily_avg',
                  'day',
                  'month',
                  'item',
                  'store',],
                 axis=1, 
                 inplace=True)
    
# Features Scaling (except sales)
sales_series, id_series = train['sales'], test['id']
# Features Scaling
train = (train - train.mean()) / train.std()
test = (test - test.mean()) / test.std()
# Retrieve actual Sales values and ID
train['sales'] = sales_series
test['id'] = id_series

# Training Data
X_train = train.drop('sales', axis=1).dropna()
y_train = train['sales']
# Test Data
test.sort_values(by=['id'], inplace=True)
X_test = test.drop('id', axis=1)
#df = train
df_train = train.copy()

# Train Test Split
X_train , X_test ,y_train, y_test = train_test_split(df_train.drop('sales',axis=1),df_train.pop('sales'), random_state=123, test_size=0.2)

# XGB Model
matrix_train = xgb.DMatrix(X_train, label = y_train)
matrix_test = xgb.DMatrix(X_test, label = y_test)

# Run XGB 
model = xgb.train(params={'objective':'reg:squarederror','eval_metric':'mae'}
                ,dtrain = matrix_train, num_boost_round = 500, 
                early_stopping_rounds = 20, evals = [(matrix_test,'test')],)


# %%
# Feature Importance Plot:
# XGBoost provides a built-in method to plot feature importance.
# You can use the plot_importance function from xgboost.

# Plot feature importance
plot_importance(model, importance_type='weight')  # You can also use 'gain' or 'cover'
plt.title('Feature Importance')
plt.show()


# %%
# Training vs Validation Error:
# You can track the training and validation errors during
# training by using the evaluation log (evals_result).

# Add evals_result to track training and validation metrics
evals_result = {}
model = xgb.train(params={'objective': 'reg:squarederror', 'eval_metric': 'mae'},
                  dtrain=matrix_train, num_boost_round=500,
                  early_stopping_rounds=20, evals=[(matrix_test, 'test')],
                  evals_result=evals_result)

# Plot training and validation error
epochs = len(evals_result['test']['mae'])
x_axis = range(0, epochs)

plt.plot(x_axis, evals_result['test']['mae'], label='Test')
plt.xlabel('Boosting Rounds')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Training vs Validation Error')
plt.legend()
plt.show()


# %%
# Predicted vs Actual Values:
# You can compare the predicted values with the 
# actual values to evaluate the model's performance.

# Predict on the test set
y_pred = model.predict(matrix_test)

# Plot predicted vs actual values
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Predicted vs Actual Sales')
plt.show()
# %%
# Residual Plot:
# A residual plot helps visualize the errors
# between predicted and actual values.

# Calculate residuals
residuals = y_test - y_pred

# Plot residuals
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Sales')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()


# %%
# Distribution of Errors:
# You can plot the distribution of residuals
# to check if they are normally distributed.

import seaborn as sns

# Plot distribution of residuals
sns.histplot(residuals, kde=True, bins=30)
plt.xlabel('Residuals')
plt.title('Distribution of Residuals')
plt.show()


# %%
# Learning Curve:
# You can plot the learning curve to see how the model's
# performance changes with the number of boosting rounds.

# Extract learning curve data

train_mae = evals_result['test']['mae']

# Plot learning curve
plt.plot(train_mae, label='Test MAE')
plt.xlabel('Boosting Rounds')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Learning Curve')
plt.legend()
plt.show()


# %%
model
# %%
