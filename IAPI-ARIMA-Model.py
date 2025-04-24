import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# Load the Excel file
df = pd.read_excel("Hackaton DB Final 04.21.xlsx", sheet_name="Supply_Demand")

# Filter only the data where Attribute == 'EffectiveDemand'
df_demand = df[df['Attribute'] == 'EffectiveDemand']

# Extract the time series
series = {}
for key in ["21A", "22B", "23C"]:
    row = df_demand[df_demand['Product ID'] == key]
    if not row.empty:
        data = row.drop(columns=['Product ID', 'Attribute']).values.flatten()
        series[key] = data
    else:
        print(f"{key} not found with Attribute='EffectiveDemand'")

# ARIMA analysis function
def analyze_series(name, data):
    print(f"\n== Analysis for {name} ==")
    ts = pd.Series(data, index=range(len(data)))
    ts_log = np.log(ts + 1)

    # ARIMA order selection - many combinations were considered
    # order = (0, 0, 0)
    # order = (0, 0, 1)
    # order = (0, 0, 2)
    # order = (0, 1, 0)
    # order = (0, 1, 1)
    # order = (0, 1, 2)
    # order = (0, 2, 0)
    # order = (0, 2, 1)
    # order = (0, 2, 2)  # Marked as unsuitable but still listed
    # order = (1, 0, 0)
    # order = (1, 0, 1)
    # order = (1, 0, 2)
    # order = (1, 1, 0)
    # order = (1, 1, 1)
    # order = (1, 1, 2)
    # order = (1, 2, 0)
    # order = (1, 2, 1)
    # order = (1, 2, 2)
    # order = (2, 0, 0)
    # order = (2, 0, 1)
    # order = (2, 1, 0)
    # order = (2, 1, 1)
    # order = (2, 1, 2)
    # order = (2, 2, 0)
    # order = (2, 2, 1)
    # order = (2, 2, 2)

    train_log = ts_log[:-5]
    test_log = ts_log[-5:]
    test_real = ts[-5:]

    order = (2, 0, 2)
    model = ARIMA(train_log, order=order)
    model_fit = model.fit()
    pred_log = model_fit.forecast(steps=5)
    pred = np.exp(pred_log) - 1

    results = pd.DataFrame({
        'Actual': test_real,
        'Prediction': pred.round(0),
        'Difference (%)': ((test_real.values - pred.values) / test_real.values * 100).round(2)
    })

    print("\nResults:\n", results)
    print("\nRMSE:", np.sqrt(mean_squared_error(test_real, pred)))
    print("MAE:", mean_absolute_error(test_real, pred))
    print("MAPE:", np.mean(np.abs((test_real - pred) / test_real)) * 100, "%")

    print("\nCross-Validation (TimeSeriesSplit, 3 folds):")
    tscv = TimeSeriesSplit(n_splits=3)
    for i, (train_idx, test_idx) in enumerate(tscv.split(ts_log)):
        train_cv = ts_log.iloc[train_idx]
        test_cv = ts_log.iloc[test_idx]
        try:
            model_cv = ARIMA(train_cv, order=order).fit()
            pred_cv_log = model_cv.forecast(steps=len(test_cv))
            pred_cv = np.exp(pred_cv_log) - 1
            real_cv = np.exp(test_cv) - 1
            mape = np.mean(np.abs((real_cv - pred_cv) / real_cv)) * 100
            print(f"Fold {i+1} - MAPE: {mape:.2f}%")
        except Exception as e:
            print(f"Fold {i+1} - Error: {e}")

    plt.figure(figsize=(10, 5))
    plt.plot(ts, label="Actual", marker='o')
    plt.plot(range(len(ts)-5, len(ts)), pred, label="Prediction", linestyle='--', marker='x', color='orange')
    plt.title(f"Series {name} - ARIMA{order} with log-transformation")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Run analysis for each series
for name, data in series.items():
    analyze_series(name, data)
