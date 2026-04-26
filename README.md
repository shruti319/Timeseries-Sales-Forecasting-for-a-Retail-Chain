# 🛒 Time Series Sales Forecasting for a Retail Chain

Forecasting monthly retail sales using classical and modern time series models — ARIMA, SARIMA, and Facebook Prophet — with full model comparison, evaluation, and an interactive Streamlit dashboard.

🔗 **Live App:** [Click here to try it](https://timeseries-sales-forecasting-for-a-retail-chain-eflcjxtzmjb7iq.streamlit.app/)

---

## 📌 Problem Statement

Retail businesses rely on accurate sales forecasts to manage inventory, staffing, and revenue planning. This project applies time series analysis techniques to historical sales data from a retail superstore to predict future monthly sales.

---

## 📂 Dataset

- **Source:** Sample Superstore dataset (commonly used for retail analytics practice)
- **Features used:** `Order Date`, `Sales`
- **Granularity:** Resampled to monthly aggregates for forecasting

---

## 📁 Repository Structure

```
├── app.py                                              ← Streamlit web app
├── Timeseries_Sales_Forecasting_for_Retail_Chain.ipynb ← Full analysis notebook
├── Sample_superstore.csv                               ← Dataset
├── requirements.txt                                    ← Dependencies
└── README.md
```

---

## 🔍 Project Workflow

1. **Data Loading & Exploration** — Shape, data types, null/duplicate checks
2. **Preprocessing** — Date parsing, resampling to daily / monthly / yearly aggregates
3. **Data Visualization** — Trend comparison, yearly bar charts, monthly distribution, seasonality by month
4. **Time Series Decomposition** — Additive decomposition into Trend, Seasonality, and Residual components
5. **Stationarity Test** — Augmented Dickey-Fuller (ADF) test
6. **ACF & PACF Analysis** — Used to determine ARIMA hyperparameters (p, d, q)
7. **Model Training & Forecasting**
   - ARIMA (1,1,1)
   - SARIMA (1,1,1)(1,1,1,12)
   - Facebook Prophet (auto-tuned)
8. **Model Evaluation** — MAE, MSE, RMSE comparison across all three models
9. **Visual Comparison** — Actual vs Predicted plot with Prophet confidence intervals

---

## 📊 Model Results Summary

| Model   | Strength |
|---------|----------|
| ARIMA   | Baseline; struggled to capture strong seasonality |
| SARIMA  | Better at seasonal patterns; improved accuracy |
| Prophet | Best overall; automatically handled trend + seasonality |

> Prophet achieved the best balance of low errors and good seasonal fit without manual parameter tuning.

---

## 🖥️ Streamlit App Features

- Upload your own CSV dataset directly in the browser
- View sales trend, yearly breakdown, and seasonality charts
- Interactive time series decomposition
- Choose a model from a dropdown — ARIMA, SARIMA, Prophet, or all three
- Adjust forecast horizon using a slider (3–24 months)
- Side-by-side MAE and RMSE metrics for model comparison
- Actual vs Predicted chart on the test set
- Future forecast chart with Prophet confidence intervals

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3 |
| Data Handling | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Time Series | Statsmodels (ARIMA, SARIMAX, seasonal_decompose, ADF, ACF/PACF) |
| Forecasting | Facebook Prophet |
| Evaluation | Scikit-learn (MAE, MSE, RMSE) |
| Web App | Streamlit |

---

## 🚀 Run Locally

1. Clone this repository
   ```bash
   git clone https://github.com/your-username/timeseries-sales-forecasting.git
   cd timeseries-sales-forecasting
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app
   ```bash
   streamlit run app.py
   ```

4. To explore the full analysis, open the notebook
   ```bash
   jupyter notebook Timeseries_Sales_Forecasting_for_Retail_Chain.ipynb
   ```

---

## 💡 Key Learnings

- Monthly aggregation reduces noise and reveals seasonality better than daily data
- ARIMA alone is insufficient for data with strong seasonal patterns
- SARIMA's seasonal order `(1,1,1,12)` significantly improved forecast accuracy
- Prophet handles seasonality automatically and is robust with minimal configuration
- Model selection should be driven by evaluation metrics *and* visual inspection of fit

---

## 👩‍💻 Author

**Shruti Bommagani**
B.Sc. Data Science | K.E.S. B K Shroff College, Mumbai
📧 shrutibommagani319@gmail.com
