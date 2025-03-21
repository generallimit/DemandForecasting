# XGBoost Sales Forecasting Model

This project utilizes an XGBoost model to forecast sales based on historical data. The Python script `xGBoost_SalesForecasting.py` performs data preprocessing, feature engineering, model training, and evaluation to predict future sales.

## Project Overview

The primary goal of this project is to build a predictive model that can accurately forecast sales for a retail store. This is achieved through the following steps:

1.  **Data Loading and Preparation:**
    *   Reads training and testing data from CSV files (`train.csv` and `test.csv`).
    *   Extracts date-related features (year, month, day, day of the year, day of the week, week of the year) from the 'date' column.
    *   Creates an additional feature based on the interaction of the day of the year and the year.

2.  **Feature Engineering:**
    *   Calculates daily and monthly average sales per item and store.
    *   Merges these average sales into the training and testing datasets as new features.
    *   Computes a rolling mean of sales over a 10-day window for each item.
    *   Shifts the rolling mean by 90 days to create a lagged rolling mean feature.

3.  **Data Preprocessing:**
    *   Drops highly correlated or redundant features (e.g., `dayofyear`, `weekofyear`, `daily_avg`, `day`, `month`, `item`, `store`).
    *   Scales numerical features using standardization (mean = 0, standard deviation = 1).
    *   Splits the training dataset into training and validation sets.

4.  **Model Training:**
    *   Trains an XGBoost regression model using the training data.
    *   Uses Mean Absolute Error (MAE) as the evaluation metric.
    *   Implements early stopping to prevent overfitting.

5.  **Model Evaluation:**
    *   **Feature Importance:** Visualizes the importance of each feature in the model's predictions.
    *   **Training vs. Validation Error:** Plots the MAE for the validation set across boosting rounds to monitor model performance and prevent overfitting.
    *   **Predicted vs. Actual Values:** Generates a scatter plot to compare the model's predictions against actual sales values.
    *   **Residual Plot:** Creates a residual plot to visualize the difference between predicted and actual sales and to check for any patterns in the errors.
    *   **Distribution of Errors:** Shows the distribution of the residuals using a histogram, which can indicate if the errors are normally distributed.
    *   **Learning Curve:** Plots the training and test errors over the number of boosting rounds to assess the model's learning progress.

## File Structure

*   `xGBoost_SalesForecasting.py`: The main Python script containing the entire sales forecasting pipeline.
*   `train.csv`: Input file containing the historical sales data for training the model.
*   `test.csv`: Input file containing the data for which sales need to be predicted.

## Libraries Used

*   `pandas`: Data manipulation and analysis.
*   `numpy`: Numerical operations.
*   `xgboost`: Gradient boosting machine implementation.
*   `datetime`: Date and time operations.
*   `sklearn.model_selection`: Data splitting.
*   `matplotlib.pyplot`: Data visualization.
*   `seaborn`: Enhanced data visualization.

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install pandas numpy xgboost scikit-learn matplotlib seaborn
    ```
2.  **Download Data:**
    Make sure that `train.csv` and `test.csv` are in the correct directory specified by the python script. The current script is expecting them in: `/home/jefferyp/Documents/projects/Retail_MLandRegressionTest/DemandForecasting/demand-forecasting-kernels-only/`
3.  **Execute the script:**
    ```bash
    python xGBoost_SalesForecasting.py
    ```
    The script will output several plots displaying model performance.

## Key Improvements and Considerations

*   **Feature Engineering:** The script includes a variety of date-based features and rolling mean features, which can capture trends and seasonality in the data.
*   **Model Evaluation:** A wide range of evaluation plots are included to help diagnose the models performance.
*   **Preprocessing:** Correctly standardizes the data.
*   **Early stopping:** Will prevent overfitting.
* **Data Paths:** You will need to make sure the paths to the input files are correct.

## Potential Future Enhancements

*   **Hyperparameter Tuning:** Optimize model parameters for improved performance.
*   **More Sophisticated Feature Engineering:** Explore more advanced feature engineering techniques, such as time series decomposition.
*   **Cross-Validation:** Use k-fold cross-validation for more robust model evaluation.
*   **Alternative Models:** Compare the XGBoost model with other time series forecasting methods.
*   **Deployment:** Deploy the model as a service or integrate it into a larger application.
* **Add in missing files:** Add in the `train.csv` and `test.csv` files so that the whole project can run from the start.
* **Clearer directory:** Make the directory less specific. Right now the paths are very specific.
