# Sales Forecasting System With Time Series

## Project Overview

This project provides a time series forecasting system that predicts future sales demand for store items. The system leverages popular forecasting models including Prophet, ARIMA, and Random Forest, wrapped in an interactive Streamlit web application.

Accurate sales forecasting helps reduce overstocking and understocking, optimizing inventories and improving business decisions.

## Features

- Data exploration with time series visualization and heatmaps
- Advanced feature engineering including lags, rolling means, and date attributes
- Forecasting models:
  - Prophet (handles trend and seasonality)
  - ARIMA (classic statistical model)
  - Random Forest with multi-step recursive forecasting
- Model evaluation (MAE, RMSE) on test data
- Export forecasts as CSV files
- User-friendly Streamlit interface with configurable inputs

## Project Structure

```
.
├── app.py                  # Main Streamlit application script
├── data/
│   ├── train.csv           # Training dataset (Kaggle competition file)
│   └── forecast_store1_item1.csv  # Example forecast output CSV
├── notebook/
│   └── Sales Forecast System With Time Series.ipynb  # Experimentation notebook
├── src/
│   ├── notebook_api.py     # Helper functions for data processing and modeling
│   └── __pycache__/        # Python cache files
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Installation and Setup

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/sales-forecasting-system.git
   cd sales-forecasting-system
   ```

2. Install required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Download the `train.csv` dataset from the Kaggle Store Item Demand Forecasting Challenge:

   [Kaggle Data Link](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data)

   Place the file inside the `data/` directory.

4. Launch the Streamlit application:

   ```
   streamlit run app.py
   ```

## Usage

- Use the sidebar to select the store, item, forecasting model, forecast horizon, and evaluation period.
- Explore historical sales trends and feature summaries.
- Run forecasts and view model predictions with associated error metrics.
- Download the forecast results as CSV files.

## Dependencies

- Python 3.8 or higher
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- prophet (fbprophet)
- statsmodels
- streamlit

Refer to `requirements.txt` for exact package versions.

## Deployment Notes

This application can be deployed on:

- Streamlit Community Cloud
- Heroku
- AWS EC2 (via Docker)
- Other Python-friendly cloud platforms

Ensure all dependencies are installed in the deployment environment. Update the `data/train.csv` file path if necessary.

## Acknowledgements

Data sourced from the Kaggle Store Item Demand Forecasting Challenge.

## License

This project is for educational and demonstrational purposes.

