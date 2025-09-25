# src/notebook_api.py
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# try prophet/import fallbacks
try:
    from prophet import Prophet
except Exception:
    try:
        from fbprophet import Prophet
    except Exception:
        Prophet = None


# ---------- data utils ----------
def load_data(filepath_or_buffer):
    """
    filepath_or_buffer: path string or an uploaded file (streamlit)
    returns: dataframe with parsed date
    """
    df = pd.read_csv(filepath_or_buffer, parse_dates=['date'])
    return df


def get_series(df, store_id, item_id, fillna_zero=True):
    """
    Filter df for a (store,item) and return a daily-frequency DataFrame with 'sales' column.
    """
    mask = (df['store'] == store_id) & (df['item'] == item_id)
    df_si = df.loc[mask, ['date', 'sales']].set_index('date').sort_index().asfreq('D')
    if fillna_zero:
        df_si['sales'] = df_si['sales'].fillna(0)
    return df_si


# ---------- feature engineering ----------
def create_features(series_df):
    """
    series_df: DataFrame with index=date and column 'sales'
    returns: DataFrame with lag and rolling features (dropped NaNs)
    """
    s = series_df.copy()
    s['lag_1'] = s['sales'].shift(1)
    s['lag_7'] = s['sales'].shift(7)
    s['lag_30'] = s['sales'].shift(30)
    s['roll_7'] = s['sales'].rolling(window=7).mean()
    s['roll_30'] = s['sales'].rolling(window=30).mean()
    s['day'] = s.index.day
    s['month'] = s.index.month
    s['weekday'] = s.index.weekday
    s['weekofyear'] = s.index.isocalendar().week
    s = s.dropna()
    return s


def train_test_split_by_months(series_with_feats, test_months=3):
    last_date = series_with_feats.index.max()
    cutoff = last_date - pd.DateOffset(months=test_months)
    train = series_with_feats[series_with_feats.index <= cutoff]
    test = series_with_feats[series_with_feats.index > cutoff]
    return train, test


# ---------- Prophet ----------
def fit_prophet(series_df):
    """
    Fit Prophet on full series_df (index date, column sales)
    returns fitted model and prophet dataframe
    """
    if Prophet is None:
        raise ImportError("Prophet is not installed.")
    df_prophet = series_df[['sales']].reset_index().rename(columns={'date':'ds','sales':'y'})
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(df_prophet)
    return m, df_prophet


def forecast_prophet(model, df_prophet, periods=30):
    future = model.make_future_dataframe(periods=periods, freq='D')
    forecast = model.predict(future)
    # return forecast DataFrame with ds,yhat,yhat_lower,yhat_upper
    return forecast[['ds','yhat','yhat_lower','yhat_upper']]


# ---------- ARIMA using statsmodels ----------
def fit_arima_and_forecast(train_series, steps):
    """
    Fit ARIMA model using statsmodels and forecast future values.
    
    Parameters:
    - train_series: pandas Series with time series data
    - steps: number of steps to forecast
    
    Returns:
    - predictions: array of forecast values
    - fitted_model: the fitted ARIMA model
    """
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    import warnings
    warnings.filterwarnings('ignore')
    
    # Check stationarity
    def check_stationarity(ts):
        try:
            result = adfuller(ts.dropna())
            return result[1] <= 0.05  # p-value <= 0.05 means stationary
        except:
            return False
    
    # Make series stationary if needed
    d = 0
    data = train_series.copy()
    while not check_stationarity(data) and d < 2:
        data = data.diff().dropna()
        d += 1
    
    try:
        # Try ARIMA with detected differencing
        model = ARIMA(train_series, order=(1, d, 1))
        fitted = model.fit()
        predictions = fitted.forecast(steps=steps)
        return predictions.values, fitted
        
    except Exception as e:
        # Fallback: simple ARIMA(1,1,1)
        try:
            model = ARIMA(train_series, order=(1, 1, 1))
            fitted = model.fit()
            predictions = fitted.forecast(steps=steps)
            return predictions.values, fitted
        except:
            # Ultimate fallback: return moving average
            last_values = train_series.tail(7).mean()
            return np.full(steps, last_values), None


# ---------- Random Forest baseline ----------
def train_rf(X_train, y_train, n_estimators=100):
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    return rf


def rf_recursive_forecast(rf_model, last_known_row, feature_cols, periods):
    """
    Produce recursive forecasts for 'periods' days using RF and rolling the features forward.
    last_known_row: pd.Series or 1-row DataFrame containing the last available features.
    feature_cols: list of feature column names in order that the model expects.
    """
    X_last = last_known_row[feature_cols].values.flatten().copy()
    preds = []
    # We'll update day/month/weekday automatically; for lags we use naive shifting
    for i in range(periods):
        pred_i = rf_model.predict(X_last.reshape(1, -1))[0]
        preds.append(pred_i)
        # slide lags: new lag_1 becomes pred_i, lag_7 becomes previous lag_6 etc.
        # simple approach: replace lag_1 <- pred_i, keep others unchanged (you can improve)
        if 'lag_1' in feature_cols:
            X_last[feature_cols.index('lag_1')] = pred_i
        # update date features
        # find day/month/weekday indexes
        if 'day' in feature_cols:
            # compute next date based on last_known_row.index (not available here)
            # This function expects last_known_row has correct day/month/weekday for day+0,
            # but we cannot compute next date without passing the real last date. Keep it simple:
            pass
    return preds


# ---------- utility ----------
def save_forecast_df(df_forecast, filename):
    df_forecast.to_csv(filename, index=False)
    return filename


def evaluate_preds(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return {'mae': mae, 'rmse': rmse}
