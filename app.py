import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Time Series Sales Forecasting — Retail Chain")
st.markdown("Upload your dataset and explore sales forecasts using ARIMA, SARIMA, and Prophet.")

st.divider()

# ── STEP 1: FILE UPLOAD ──────────────────────────────────────
st.header("📂 Step 1: Upload Dataset")
uploaded_file = st.file_uploader("Upload Sample_superstore.csv", type=["csv"])

if uploaded_file is None:
    st.info("👆 Please upload your CSV file to get started.")
    st.stop()

# ── LOAD & PREPROCESS ────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df_imp = df[['Order Date', 'Sales']].copy()
    df_imp['Order Date'] = pd.to_datetime(df_imp['Order Date'], dayfirst=True)
    df_imp = df_imp.set_index('Order Date').sort_index()
    monthly_sales = df_imp['Sales'].resample('MS').sum()
    yearly_sales  = df_imp['Sales'].resample('YS').sum()
    df_imp['Month'] = df_imp.index.month
    avg_monthly = df_imp.groupby('Month')['Sales'].sum()
    return monthly_sales, yearly_sales, avg_monthly

monthly_sales, yearly_sales, avg_monthly = load_data(uploaded_file)

st.success(f"✅ Dataset loaded! Monthly data spans **{monthly_sales.index[0].strftime('%b %Y')}** to **{monthly_sales.index[-1].strftime('%b %Y')}** ({len(monthly_sales)} months)")

st.divider()

# ── STEP 2: VISUALISATIONS ───────────────────────────────────
st.header("📊 Step 2: Sales Overview")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Monthly Sales Trend")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(monthly_sales, color='steelblue', linewidth=2)
    ax.set_xlabel("Date"); ax.set_ylabel("Sales")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Total Sales by Year")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(yearly_sales.index.year, yearly_sales.values, color='navy')
    ax.set_xlabel("Year"); ax.set_ylabel("Total Sales")
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.subheader("Average Sales by Month (Seasonality)")
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(avg_monthly.index, avg_monthly.values, color='teal')
ax.set_xlabel("Month"); ax.set_ylabel("Total Sales")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.divider()

# ── STEP 3: DECOMPOSITION ────────────────────────────────────
st.header("🔍 Step 3: Time Series Decomposition")
result = seasonal_decompose(monthly_sales, model='additive', period=12)
fig, axes = plt.subplots(4, 1, figsize=(12, 8))
for ax, component, label in zip(
    axes,
    [result.observed, result.trend, result.seasonal, result.resid],
    ['Observed', 'Trend', 'Seasonal', 'Residual']
):
    ax.plot(component, color='steelblue')
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
fig.suptitle("Additive Decomposition", fontsize=14)
plt.tight_layout()
st.pyplot(fig)
plt.close()

adf_stat, p_value = adfuller(monthly_sales)[:2]
st.markdown(f"**ADF Test:** Statistic = `{adf_stat:.4f}`, p-value = `{p_value:.4f}` → "
            f"{'✅ Stationary' if p_value < 0.05 else '⚠️ Non-Stationary (differencing needed)'}")

st.divider()

# ── STEP 4: FORECASTING ──────────────────────────────────────
st.header("🔮 Step 4: Forecast")

col_a, col_b = st.columns(2)
with col_a:
    model_choice = st.selectbox(
        "Choose a forecasting model",
        ["ARIMA (1,1,1)", "SARIMA (1,1,1)(1,1,1,12)", "Prophet", "Compare All Three"]
    )
with col_b:
    forecast_months = st.slider("Months to forecast into the future", min_value=3, max_value=24, value=6)

run_btn = st.button("▶ Run Forecast", type="primary")

if run_btn:
    train_size = int(len(monthly_sales) * 0.8)
    train = monthly_sales.iloc[:train_size]
    test  = monthly_sales.iloc[train_size:]

    results = {}

    with st.spinner("Training model(s)... this may take a moment ⏳"):

        # ARIMA
        if model_choice in ["ARIMA (1,1,1)", "Compare All Three"]:
            arima_model  = ARIMA(train, order=(1,1,1)).fit()
            arima_pred   = arima_model.forecast(steps=len(test))
            arima_future = arima_model.forecast(steps=len(test) + forecast_months)[-forecast_months:]
            results['ARIMA'] = {
                'pred': arima_pred,
                'future': arima_future,
                'mae':  mean_absolute_error(test, arima_pred),
                'rmse': np.sqrt(mean_squared_error(test, arima_pred)),
                'color': 'red'
            }

        # SARIMA
        if model_choice in ["SARIMA (1,1,1)(1,1,1,12)", "Compare All Three"]:
            sarima_model  = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
            sarima_pred   = sarima_model.forecast(steps=len(test))
            sarima_future = sarima_model.forecast(steps=len(test) + forecast_months)[-forecast_months:]
            results['SARIMA'] = {
                'pred': sarima_pred,
                'future': sarima_future,
                'mae':  mean_absolute_error(test, sarima_pred),
                'rmse': np.sqrt(mean_squared_error(test, sarima_pred)),
                'color': 'blue'
            }

        # Prophet
        if model_choice in ["Prophet", "Compare All Three"]:
            prophet_df = monthly_sales.reset_index()
            prophet_df.columns = ['ds', 'y']
            train_prophet = prophet_df.iloc[:train_size]
            test_prophet  = prophet_df.iloc[train_size:]
            prophet_model = Prophet()
            prophet_model.fit(train_prophet)
            future_df = prophet_model.make_future_dataframe(periods=len(test) + forecast_months, freq='MS')
            forecast  = prophet_model.predict(future_df)
            prophet_test_pred   = forecast.iloc[train_size:train_size+len(test)]['yhat'].values
            prophet_future_pred = forecast.iloc[-forecast_months:]
            results['Prophet'] = {
                'pred': prophet_test_pred,
                'future_df': prophet_future_pred,
                'mae':  mean_absolute_error(test_prophet['y'], prophet_test_pred),
                'rmse': np.sqrt(mean_squared_error(test_prophet['y'], prophet_test_pred)),
                'color': 'green'
            }

    # ── METRICS ─────────────────────────────────────────────
    st.subheader("📈 Model Performance on Test Set")
    metric_cols = st.columns(len(results))
    for col, (name, r) in zip(metric_cols, results.items()):
        col.metric(label=f"{name} — MAE",  value=f"{r['mae']:,.0f}")
        col.metric(label=f"{name} — RMSE", value=f"{r['rmse']:,.0f}")

    # ── ACTUAL vs PREDICTED PLOT ─────────────────────────────
    st.subheader("Actual vs Predicted (Test Period)")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train, label='Train', color='gray', alpha=0.6)
    ax.plot(test,  label='Actual (Test)', color='black', linewidth=2)
    for name, r in results.items():
        if name != 'Prophet':
            ax.plot(test.index, r['pred'], label=f'{name} Prediction',
                    color=r['color'], linestyle='--', linewidth=1.8)
        else:
            ax.plot(test.index, r['pred'], label='Prophet Prediction',
                    color=r['color'], linestyle='--', linewidth=1.8)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── FUTURE FORECAST PLOT ─────────────────────────────────
    st.subheader(f"🔮 Future Forecast — Next {forecast_months} Months")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly_sales, label='Historical Sales', color='gray', alpha=0.7)

    import pandas.tseries.offsets as offsets
    last_date    = monthly_sales.index[-1]
    future_index = pd.date_range(start=last_date + offsets.MonthBegin(1), periods=forecast_months, freq='MS')

    for name, r in results.items():
        if name == 'Prophet':
            ax.plot(r['future_df']['ds'].values, r['future_df']['yhat'].values,
                    label=f'{name} Forecast', color=r['color'], linewidth=2, linestyle='--')
            ax.fill_between(r['future_df']['ds'].values,
                            r['future_df']['yhat_lower'].values,
                            r['future_df']['yhat_upper'].values,
                            color=r['color'], alpha=0.1, label='Prophet 95% CI')
        else:
            ax.plot(future_index, r['future'].values,
                    label=f'{name} Forecast', color=r['color'], linewidth=2, linestyle='--')

    ax.axvline(x=last_date, color='black', linestyle=':', alpha=0.5, label='Forecast Start')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.success("✅ Forecast complete!")

st.divider()
st.caption("Built by Shruti Bommagani · B.Sc. Data Science · K.E.S. B K Shroff College")
