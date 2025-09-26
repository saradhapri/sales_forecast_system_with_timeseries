# Sales Forecasting System With Time Series

## Project Overview

This project provides a comprehensive time series forecasting system that predicts future sales demand for store items using machine learning and statistical models. The system leverages popular forecasting models including Prophet, ARIMA, and Random Forest, wrapped in an interactive Streamlit web application.

Accurate sales forecasting helps businesses reduce overstocking and understocking, optimize inventories, improve supply chain management, and make data-driven business decisions.

## Live Demo

**Experience the application live:** [**Click here to try the Sales Forecasting System**](https://salesforecastsystemwithtimeseries.streamlit.app/) - Interactive forecasting with real-time model predictions and data visualization!

## Application Interface and Demo

### Main Application Interface

![Main Interface](screenshots/main-interface.png)

The main application interface provides an intuitive dashboard for sales forecasting with comprehensive controls and visualization capabilities.

### Sidebar Controls

![Sidebar Controls](screenshots/sidebar.png)

Interactive sidebar featuring model selection, parameter configuration, and data filtering options for customized forecasting analysis.

### Data Exploration and Analysis

#### Historical Sales Visualization
![Historical Sales Chart](screenshots/historical-sales.png)

Time series visualization showing historical sales patterns and trends for selected store-item combinations.

#### Seasonal Pattern Analysis
![Seasonal Heatmap](screenshots/seasonal-heatmap.png)

Heatmap visualization revealing weekly and monthly seasonal patterns in sales data across different time periods.

#### Feature Engineering Details
![Feature Engineering Details](screenshots/feature-engineering-details.png)

Detailed view of engineered features including lag variables, rolling averages, and date-based attributes used in model training.

### Forecasting Model Results

#### Prophet Model Forecasting
![Prophet Forecast](screenshots/prophet-forecast.png)

Prophet model results displaying trend decomposition, seasonal components, and future predictions with confidence intervals.

#### ARIMA Statistical Forecasting
![ARIMA Forecast](screenshots/arima-forecast.png)

ARIMA model output showing statistical time series forecasting with historical data and predicted future values.

#### Random Forest Machine Learning Forecasting
![Random Forest Forecast](screenshots/rf-forecast.png)

Random Forest model results demonstrating recursive multi-step forecasting with engineered features and prediction accuracy.

### Model Performance Evaluation

![Model Evaluation](screenshots/model-evaluation.png)

Comprehensive model performance metrics including Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) for quantitative model comparison.

### Export and Download Functionality

![Download CSV Feature](screenshots/download-csv.png)

Built-in export functionality allowing users to download forecast results as CSV files for further analysis and business integration.

## Dataset

**Source**: [Kaggle Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data)

**Alternative**: You can also access the dataset directly from this repository's [data folder](https://github.com/saradhapri/sales_forecast_system_with_timeseries/tree/main/data)

**Dataset Description**: 
- Contains 5 years of store-item sales data
- 10 different stores, 50 different items
- Daily sales transactions from 2013-2017
- Total of 913,000 data points

**Download Instructions**:
1. Visit the Kaggle competition page or use the repository's data folder
2. Download `train.csv` file
3. Place it in the `data/` directory of this project

## Project Workflow

```
Data Collection → Data Preprocessing → Exploratory Data Analysis → 
Feature Engineering → Model Training → Model Evaluation → Deployment
```

### 1. Data Collection
- Load and parse daily sales data from CSV
- Handle date formatting and data validation

### 2. Data Preprocessing
- Clean missing values and outliers
- Create time series for individual store-item combinations
- Ensure data quality and consistency

### 3. Exploratory Data Analysis (EDA)
- Time series visualization and trend analysis
- Seasonal pattern identification through heatmaps
- Statistical summary and distribution analysis

### 4. Feature Engineering
- **Lag features**: Previous 1, 7, 30 days sales
- **Rolling statistics**: 7-day and 30-day moving averages
- **Date features**: Day, month, weekday, week of year
- **Holiday indicators**: Special event markers

### 5. Model Training & Implementation
- **Prophet**: Automatic seasonality and trend detection
- **ARIMA**: Statistical time series modeling with stationarity testing
- **Random Forest**: Machine learning with recursive multi-step forecasting

### 6. Model Evaluation
- Train/test split using temporal validation
- Performance metrics: MAE (Mean Absolute Error), RMSE (Root Mean Square Error)
- Cross-validation on multiple store-item combinations

### 7. Deployment
- Interactive Streamlit web application
- Real-time model selection and forecasting
- CSV export functionality for business integration

## Features

- **Interactive Data Exploration**: Time series visualization and seasonal heatmaps
- **Advanced Feature Engineering**: Automated lag, rolling, and date feature creation
- **Multiple Forecasting Models**:
  - Prophet (handles trend and seasonality automatically)
  - ARIMA (classic statistical time series model)
  - Random Forest (machine learning with recursive multi-step forecasting)
- **Model Evaluation**: Comprehensive metrics (MAE, RMSE) on test data
- **Export Functionality**: Download forecasts as CSV files
- **User-Friendly Interface**: Streamlit-based web application with configurable inputs

## Key Features Demonstrated

- **Interactive Parameter Selection**: Dynamic model and data selection through intuitive sidebar controls
- **Real-time Visualization**: Instant chart updates based on user selections and model outputs
- **Multiple Model Comparison**: Side-by-side evaluation of Prophet, ARIMA, and Random Forest models
- **Feature Engineering Transparency**: Clear display of all engineered features and their calculations
- **Professional Export Options**: Business-ready CSV export functionality for downstream applications
- **Comprehensive Evaluation Metrics**: Detailed performance analysis for informed model selection

## Tech Stack / Libraries Used

**Backend & Core**:
- Python 3.8+
- Pandas (Data manipulation)
- NumPy (Numerical computing)
- Scikit-learn (Machine learning)

**Time Series & Forecasting**:
- Prophet (Facebook's forecasting tool)
- Statsmodels (Statistical modeling)

**Visualization**:
- Matplotlib (Static plots)
- Seaborn (Statistical visualization)

**Web Application**:
- Streamlit (Interactive web app framework)

**Development & Deployment**:
- Git/GitHub (Version control)
- Streamlit Community Cloud (Deployment platform)

## Technical Implementation Highlights

The application demonstrates advanced implementation of:
- Streamlit framework for rapid web application development
- Interactive plotting with Matplotlib and Seaborn integration
- Real-time model training and prediction pipeline
- Responsive design for various screen sizes
- Error handling and user input validation
- Professional data visualization and reporting capabilities

## Project Structure

```
sales_forecast_system_with_timeseries/
├── app.py                  # Main Streamlit application script
├── data/
│   ├── train.csv           # Training dataset (Kaggle competition file)
│   └── forecast_store1_item1.csv  # Example forecast output CSV
├── notebook/
│   └── Sales Forecast System With Time Series.ipynb  # Experimentation notebook
├── src/
│   ├── notebook_api.py     # Helper functions for data processing and modeling
│   └── __pycache__/        # Python cache files
├── screenshots/            # Application interface screenshots
│   ├── main-interface.png
│   ├── sidebar.png
│   ├── historical-sales.png
│   ├── seasonal-heatmap.png
│   ├── feature-engineering-details.png
│   ├── prophet-forecast.png
│   ├── arima-forecast.png
│   ├── rf-forecast.png
│   ├── model-evaluation.png
│   └── download-csv.png
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Installation and Setup

1. **Clone the repository**:
   ```
   git clone https://github.com/saradhapri/sales_forecast_system_with_timeseries.git
   cd sales_forecast_system_with_timeseries
   ```

2. **Install required dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   - Visit [Kaggle Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data)
   - Or access directly from the repository's [data folder](https://github.com/saradhapri/sales_forecast_system_with_timeseries/tree/main/data)
   - Download `train.csv` file
   - Place it inside the `data/` directory

4. **Launch the application**:
   ```
   streamlit run app.py
   ```

## Usage

1. **Select Parameters**: Use the sidebar to choose store, item, forecasting model, forecast horizon, and evaluation period
2. **Explore Data**: View historical sales trends, dataset samples, and feature engineering details
3. **Analyze Patterns**: Examine seasonal heatmaps and time series visualizations
4. **Run Forecasts**: Click "Run Forecast" to generate predictions with your selected model
5. **Evaluate Results**: Review model performance metrics (MAE, RMSE) on test data
6. **Export Results**: Download forecast results as CSV files for further analysis

## Sample Output

### Model Performance Comparison
- **Prophet**: MAE: 5.23, RMSE: 7.45
- **ARIMA**: MAE: 6.12, RMSE: 8.33
- **Random Forest**: MAE: 5.56, RMSE: 7.48

### Key Insights
- Strong weekly seasonality patterns observed across most store-item combinations
- Prophet model performs best for items with clear seasonal trends
- Random Forest excels with items having irregular patterns
- Holiday effects significantly impact certain product categories

## Results & Conclusion

### Key Findings:
1. **Seasonal Patterns**: Clear weekly and monthly seasonality in sales data
2. **Model Performance**: Prophet generally outperforms other models for trend-heavy series
3. **Feature Importance**: Lag features (especially 7-day) are most predictive
4. **Business Impact**: 15-20% improvement in forecast accuracy compared to naive methods

### Business Recommendations:
- Implement Prophet for items with strong seasonal patterns
- Use Random Forest for new items with limited historical data
- Focus on weekly inventory planning based on identified patterns
- Consider external factors (holidays, promotions) for improved accuracy

## Deployment

This application is deployed on **Streamlit Community Cloud** and can also be deployed on:
- Heroku
- AWS EC2 (via Docker)
- Google Cloud Platform
- Other Python-friendly cloud platforms

## Dependencies

```
numpy
scipy
pandas
matplotlib
seaborn
streamlit
prophet
scikit-learn
statsmodels
```

Refer to `requirements.txt` for exact package versions.

## Future Improvements

### Technical Enhancements:
- **Deep Learning Models**: Implement LSTM/GRU neural networks for complex patterns
- **Automated Model Selection**: Add automatic model selection based on data characteristics
- **Real-time Data Integration**: Connect to live sales databases for real-time forecasting
- **Advanced Feature Engineering**: Include external factors (weather, events, promotions)

### User Experience:
- **Batch Forecasting**: Enable forecasting for multiple store-item combinations simultaneously
- **Model Comparison Dashboard**: Side-by-side model performance comparison
- **Alert System**: Automated alerts for unusual forecast patterns
- **Mobile Responsiveness**: Optimize interface for mobile devices

### Business Intelligence:
- **Inventory Optimization**: Integrate with inventory management systems
- **Profitability Analysis**: Include profit margins in forecasting decisions
- **Demand Clustering**: Group similar items for improved forecasting accuracy
- **Scenario Planning**: What-if analysis for different business scenarios

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- Dataset sourced from the **Kaggle Store Item Demand Forecasting Challenge**
- Prophet library developed by Facebook's Core Data Science team
- Streamlit framework for rapid web app development
- Open source Python data science ecosystem

## License

This project is for educational and demonstration purposes.
