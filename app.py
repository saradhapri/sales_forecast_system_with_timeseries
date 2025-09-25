import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.notebook_api import (
    load_data, get_series, create_features, train_test_split_by_months,
    fit_prophet, forecast_prophet, fit_arima_and_forecast, train_rf, save_forecast_df
)

st.set_page_config(page_title="Sales Forecast System", layout="wide")
st.title("Sales Forecast System With Time Series")

# Sidebar about section
with st.sidebar.expander("About this app"):
    st.write("""
    Time series forecasting app using Prophet, ARIMA, and Random Forest models.
    Includes EDA, feature engineering, evaluation, and multi-day forecasting.
    Dataset: Kaggle Store Item Demand Forecasting.
    """)

# Load data
data_path = "data/train.csv"
try:
    df = load_data(data_path)
except Exception as e:
    st.error(f"Could not load data file at {data_path}. Error: {e}")
    st.stop()

if (
    df is None or
    not isinstance(df, pd.DataFrame) or
    'store' not in df.columns or
    'item' not in df.columns or
    'sales' not in df.columns
):
    st.error("Data did not load correctly, or required columns ('store', 'item', 'sales') missing.")
    st.stop()

# Sidebar controls
st.sidebar.header("Data & Options")
stores = sorted(df['store'].unique())
store = st.sidebar.selectbox("Store", stores, index=0, help="Select store")
items = sorted(df['item'].unique())
item = st.sidebar.selectbox("Item", items, index=0, help="Select item")
model_choice = st.sidebar.radio("Model", ['Prophet','ARIMA','RandomForest'], help="Choose forecasting model")
forecast_days = st.sidebar.slider("Forecast days", 7, 90, 30, 7, help="Days to forecast ahead")
test_months = st.sidebar.slider("Test months (for evaluation)", 1, 6, 3, help="Months for evaluation split")

# Preview dataset
st.subheader(f"Dataset sample for Store {store}, Item {item}")
st.dataframe(df[(df['store'] == store) & (df['item'] == item)].head())

series = get_series(df, store, item)
st.subheader(f"Historical sales for Store {store}, Item {item}")
st.line_chart(series['sales'])

with st.expander("Feature Engineering Details"):
    st.write("""
    - Lag features: 1, 7, 30 days
    - Rolling averages: 7-day, 30-day
    - Date features: day, month, weekday, holiday flags
    """)

with st.expander("Sales Heatmap by Day-of-Week vs. Month"):
    tmp = series.copy()
    tmp['dayofweek'] = tmp.index.dayofweek
    tmp['month'] = tmp.index.month
    piv = tmp.pivot_table(values='sales', index='dayofweek', columns='month', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(9,5))
    sns.heatmap(piv, annot=True, fmt=".1f", ax=ax, cmap='Blues')
    st.pyplot(fig)

# Recursive RF forecasting function
def recursive_rf_forecast(rf_model, series_feats, forecast_days):
    feature_cols = ['lag_1', 'lag_7', 'lag_30', 'roll_7', 'roll_30', 'day', 'month', 'weekday']
    df_forecast = series_feats.copy()

    for i in range(forecast_days):
        last_date = df_forecast.index[-1]
        next_date = last_date + pd.Timedelta(days=1)

        new_row = {}
        new_row['day'] = next_date.day
        new_row['month'] = next_date.month
        new_row['weekday'] = next_date.weekday()

        def get_lag(lag):
            lag_date = next_date - pd.Timedelta(days=lag)
            return df_forecast['sales'].get(lag_date, 0)

        new_row['lag_1'] = get_lag(1)
        new_row['lag_7'] = get_lag(7)
        new_row['lag_30'] = get_lag(30)

        def get_roll(window):
            past_dates = [next_date - pd.Timedelta(days=x) for x in range(1, window + 1)]
            values = [df_forecast['sales'].get(d, 0) for d in past_dates]
            if len(values) == 0:
                return 0
            else:
                return np.mean(values)

        new_row['roll_7'] = get_roll(7)
        new_row['roll_30'] = get_roll(30)

        df_new = pd.DataFrame(new_row, index=[next_date])

        pred = rf_model.predict(df_new[feature_cols])[0]
        df_new['sales'] = pred

        df_forecast = pd.concat([df_forecast, df_new])

    forecast_df = df_forecast.tail(forecast_days).reset_index()
    forecast_df.rename(columns={'index': 'ds', 'sales': 'yhat'}, inplace=True)
    return forecast_df[['ds', 'yhat']]

# Run forecast based on user choice
if st.sidebar.button("Run Forecast"):
    with st.spinner("Running forecast..."):
        series_feats = create_features(series)
        train, test = train_test_split_by_months(series_feats, test_months)

        if model_choice == 'Prophet':
            try:
                model, df_prop = fit_prophet(series)
                forecast_df = forecast_prophet(model, df_prop, periods=forecast_days)
                res = forecast_df.tail(forecast_days).copy()
                st.subheader(f"Forecast (Prophet): Store {store}, Item {item}")
                st.dataframe(res.set_index('ds'))
                fig, ax = plt.subplots(figsize=(12,5))
                ax.plot(series.index, series['sales'], label='Historical', color='blue')
                ax.plot(res['ds'], res['yhat'], '--o', color='red', label='Forecast')
                ax.fill_between(res['ds'], res['yhat_lower'], res['yhat_upper'], alpha=0.2, color='pink', label='Confidence Interval')
                ax.legend()
                st.pyplot(fig)
                if test.shape[0] > 0:
                    from sklearn.metrics import mean_absolute_error, mean_squared_error
                    y_true = test['sales']
                    y_pred = forecast_df.loc[forecast_df['ds'].isin(test.index), 'yhat']
                    if len(y_true) == len(y_pred):
                        mae = mean_absolute_error(y_true, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                        st.info(f"Test MAE: {mae:.2f}, Test RMSE: {rmse:.2f}")
                csv = res.to_csv(index=False).encode('utf-8')
                st.download_button("Download Forecast CSV", csv, f"forecast_store{store}_item{item}.csv")
            except Exception as e:
                st.error(f"Prophet failed: {e}")

        elif model_choice == 'ARIMA':
            preds, fitted = fit_arima_and_forecast(train['sales'], forecast_days)
            forecast_dates = pd.date_range(start=series.index.max() + pd.Timedelta(days=1), periods=forecast_days, freq='D')
            res = pd.DataFrame({'ds': forecast_dates, 'yhat': preds})
            st.subheader(f"Forecast (ARIMA): Store {store}, Item {item}")
            st.dataframe(res.set_index('ds'))
            fig, ax = plt.subplots(figsize=(12,5))
            ax.plot(series.index, series['sales'], label='Historical', color='blue')
            ax.plot(res['ds'], res['yhat'], '--o', color='red', label='ARIMA Forecast')
            ax.legend()
            st.pyplot(fig)
            
            # Fix evaluation: only evaluate if we have test data of matching length
            if test.shape[0] > 0:
                from sklearn.metrics import mean_absolute_error, mean_squared_error
                # Fit ARIMA on train data and predict on test period
                test_preds, _ = fit_arima_and_forecast(train['sales'], len(test))
                if len(test_preds) == len(test):
                    y_true = test['sales']
                    y_pred = test_preds[:len(test)]  # Ensure matching length
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    st.info(f"Test MAE: {mae:.2f}, Test RMSE: {rmse:.2f}")
                else:
                    st.info("Test evaluation skipped due to length mismatch")
            
            csv = res.to_csv(index=False).encode('utf-8')
            st.download_button("Download Forecast CSV", csv, f"arima_forecast_store{store}_item{item}.csv")

        else:  # RandomForest with recursive forecast
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            from math import sqrt
            feature_cols = ['lag_1', 'lag_7', 'lag_30', 'roll_7', 'roll_30', 'day', 'month', 'weekday']
            X_train, y_train = train[feature_cols], train['sales']
            X_test, y_test = test[feature_cols], test['sales']

            st.subheader("Training Random Forest model...")
            rf = train_rf(X_train, y_train, n_estimators=200)

            # Test set evaluation
            preds_test = rf.predict(X_test)
            evals = {'MAE': mean_absolute_error(y_test, preds_test), 'RMSE': sqrt(mean_squared_error(y_test, preds_test))}
            st.subheader("Evaluation on Test Set")
            st.write(evals)

            # Recursive multi-day forecasting
            st.subheader(f"Forecasting next {forecast_days} days with Random Forest...")
            forecast_df = recursive_rf_forecast(rf, train, forecast_days)

            fig, ax = plt.subplots(figsize=(12,5))
            ax.plot(series.index, series['sales'], label='Historical', color='blue')
            ax.plot(forecast_df['ds'], forecast_df['yhat'], '--o', color='red', label='RF Forecast')
            ax.legend()
            st.pyplot(fig)

            st.dataframe(forecast_df.set_index('ds'))
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download RF Forecast CSV", csv, f"rf_forecast_store{store}_item{item}.csv")
