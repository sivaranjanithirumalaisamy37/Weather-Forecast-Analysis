import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_squared_error, r2_score
)
from scipy.stats import norm

# ---------------------------------------------------
# LOAD DATASET SAFELY
# ---------------------------------------------------
file_path = "C:\\Users\\Siva Ranjani\\Downloads\\archive (8).zip"

def load_weather_file(path):
    encodings = ["utf-8", "latin1", "ISO-8859-1", "cp1252"]

    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="replace") as f:
                sample = f.read(5000)

            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
                sep = dialect.delimiter
            except:
                sep = None

            if sep is not None:
                try:
                    df = pd.read_csv(path, encoding=enc, sep=sep, engine="python")
                    if df.shape[1] > 1:
                        print(f"Loaded using encoding={enc}, separator='{sep}'")
                        return df
                except:
                    pass

            for trial_sep in [",", ";", "\t", "|"]:
                try:
                    df = pd.read_csv(path, encoding=enc, sep=trial_sep, engine="python")
                    if df.shape[1] > 1:
                        print(f"Loaded using encoding={enc}, separator='{trial_sep}'")
                        return df
                except:
                    pass

        except:
            pass

    raise ValueError("Could not read the file properly. It may not be a valid CSV.")

df = load_weather_file(file_path)

print("========== RAW DATASET OVERVIEW ==========")
print(df.head())

print("\n========== RAW COLUMN NAMES ==========")
print(df.columns.tolist())

# ---------------------------------------------------
# STANDARDIZE COLUMN NAMES
# ---------------------------------------------------
df.columns = df.columns.str.strip().str.replace(" ", "_")

rename_map = {
    "temperature": "Temperature",
    "humidity": "Humidity",
    "wind_speed": "Wind_Speed",
    "cloud_cover": "Cloud_Cover",
    "pressure": "Pressure",
    "rain": "Rain"
}

df.columns = [col.strip() for col in df.columns]
df = df.rename(columns={c: rename_map.get(c.lower(), c) for c in df.columns})

print("\n========== CLEANED COLUMN NAMES ==========")
print(df.columns.tolist())

required_cols = ['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure', 'Rain']

missing = [col for col in required_cols if col not in df.columns]
if missing:
    print("\nMissing required columns:", missing)
    print("Available columns:", df.columns.tolist())
    raise KeyError("Required columns not found correctly in the dataset.")

df = df[required_cols].copy()

# ---------------------------------------------------
# BASIC PREPROCESSING
# ---------------------------------------------------
numeric_cols = ['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].mean())

df['Rain'] = df['Rain'].astype(str).str.strip().str.lower()

df['Rain_Binary'] = df['Rain'].map({
    'rain': 1,
    'no rain': 0
})

if df['Rain_Binary'].isnull().any():
    print("\nUnrecognized Rain values found:")
    print(df['Rain'].unique())
    df['Rain_Binary'] = df['Rain_Binary'].fillna(0)

print("\n========== CLEANED DATASET ==========")
print(df.head())

print("\n========== DATASET INFO ==========")
print(df.info())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# ===================================================
# EXPERIMENT 1: CORRELATION MATRIX
# ===================================================
print("\n\n===== EXPERIMENT 1: CORRELATION MATRIX =====")
corr_features = ['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure', 'Rain_Binary']

correlation_matrix = np.corrcoef(df[corr_features].to_numpy(), rowvar=False)

print("Correlation Matrix:")
print(pd.DataFrame(correlation_matrix, index=corr_features, columns=corr_features))

plt.figure(figsize=(10, 7))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    xticklabels=corr_features,
    yticklabels=corr_features,
    linewidths=0.5
)
plt.title("Weather Features Correlation Matrix")
plt.tight_layout()
plt.show()

# ===================================================
# EXPERIMENT 2: LOGISTIC REGRESSION
# ===================================================
print("\n\n===== EXPERIMENT 2: LOGISTIC REGRESSION =====")

X = df[['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']]
y = df['Rain_Binary']

print("Unique classes in Rain_Binary:", y.unique())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_scaled, y_train)

y_pred = log_model.predict(X_test_scaled)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall   :", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score :", f1_score(y_test, y_pred, zero_division=0))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Rain", "Rain"],
    yticklabels=["No Rain", "Rain"]
)
plt.title("Confusion Matrix - Rain Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ===================================================
# EXPERIMENT 3: LINEAR REGRESSION
# ===================================================
print("\n\n===== EXPERIMENT 3: LINEAR REGRESSION =====")

X_lin = df[['Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure', 'Rain_Binary']]
y_lin = df['Temperature']

X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
    X_lin, y_lin, test_size=0.2, random_state=42
)

lin_model = LinearRegression()
lin_model.fit(X_train_lin, y_train_lin)
y_pred_lin = lin_model.predict(X_test_lin)

mse = mean_squared_error(y_test_lin, y_pred_lin)
r2 = r2_score(y_test_lin, y_pred_lin)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

plt.figure(figsize=(8, 5))
plt.scatter(y_test_lin, y_pred_lin, alpha=0.7)
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Linear Regression: Actual vs Predicted Temperature")
plt.grid(True)
plt.tight_layout()
plt.show()

# ===================================================
# EXPERIMENT 4: RANDOM SAMPLING
# ===================================================
print("\n\n===== EXPERIMENT 4: RANDOM SAMPLING =====")
sample_25 = df.sample(frac=0.25, random_state=42)
print(sample_25.head(10))

# ===================================================
# EXPERIMENT 5: Z-TEST
# ===================================================
print("\n\n===== EXPERIMENT 5: Z-TEST =====")
sample_temp = df["Temperature"].sample(n=min(40, len(df)), random_state=42).to_numpy()

sample_mean = np.mean(sample_temp)
sample_std = np.std(sample_temp, ddof=1)
sample_size = len(sample_temp)
null_mean = 25
alpha = 0.05

standard_error = sample_std / np.sqrt(sample_size)
z_score = (sample_mean - null_mean) / standard_error
critical_z = norm.ppf(1 - alpha)

decision = "Reject H0" if z_score > critical_z else "Fail to Reject H0"

print(f"Sample Mean      : {sample_mean:.2f}")
print(f"Sample Std Dev   : {sample_std:.2f}")
print(f"Z Score          : {z_score:.2f}")
print(f"Critical Z Value : {critical_z:.2f}")
print(f"Decision         : {decision}")

# ===================================================
# EXPERIMENT 6: NUMPY OPERATIONS
# ===================================================
print("\n\n===== EXPERIMENT 6: NUMPY OPERATIONS =====")
rain_array = df["Rain_Binary"].to_numpy()

print("Rain Array:", rain_array)
print("Shape :", rain_array.shape)
print("Mean  :", np.mean(rain_array))
print("Sum   :", np.sum(rain_array))
print("Prod  :", np.prod(rain_array + 1))

reshaped = rain_array.reshape(-1, 1)
transposed = reshaped.T

print("Reshaped Array:\n", reshaped)
print("Transposed Array:\n", transposed)

# ===================================================
# EXPERIMENT 7: CLEANING ARRAY
# ===================================================
print("\n\n===== EXPERIMENT 7: CLEANING ARRAY =====")

mixed_array = np.array([
    [25, 80, 5.2],
    [30, np.nan, 7.1],
    [28, 75, "invalid"],
    [32, 60, 4.8],
    [29, np.nan, 6.3]
], dtype=object)

print("Original Mixed Array:\n", mixed_array)

clean_rows = []
for row in mixed_array:
    try:
        numeric_row = [float(x) for x in row]
        clean_rows.append(numeric_row)
    except Exception:
        pass

clean_array = np.array(clean_rows, dtype=float)
print("\nAfter Removing Non-Numeric Rows:\n", clean_array)

col_means = np.nanmean(clean_array, axis=0)
inds = np.where(np.isnan(clean_array))
clean_array[inds] = np.take(col_means, inds[1])

print("\nAfter Replacing NaN with Column Mean:\n", clean_array)

row_to_check = np.array([32.0, 60.0, 4.8])
row_exists = np.any(np.all(clean_array == row_to_check, axis=1))
print("\nDoes row exist?", "Yes" if row_exists else "No")

# ===================================================
# EXPERIMENT 8: WEATHER SIMULATION
# ===================================================
print("\n\n===== EXPERIMENT 8: WEATHER SIMULATION =====")

temp_series = df["Temperature"].to_numpy()
changes = np.diff(temp_series)

mean_change = np.mean(changes)
std_change = np.std(changes)

future_days = 30
simulated_temp = np.zeros(future_days)
simulated_temp[0] = temp_series[-1]

for i in range(1, future_days):
    shock = np.random.normal(mean_change, std_change)
    simulated_temp[i] = simulated_temp[i - 1] + shock

print("Simulated Next 30 Days Temperature:")
print(simulated_temp)

plt.figure(figsize=(10, 5))
plt.plot(range(1, future_days + 1), simulated_temp, marker='o')
plt.title("Simulated Future Temperature (30 Days)")
plt.xlabel("Future Day")
plt.ylabel("Temperature")
plt.grid(True)
plt.tight_layout()
plt.show()

# ===================================================
# EXPERIMENT 9: WEATHER TREND ANALYSIS
# ===================================================
print("\n\n===== EXPERIMENT 9: WEATHER TREND ANALYSIS =====")

feature_means = df[corr_features].mean()
print("Average of Weather Features:\n", feature_means)

highest_temp_row = df.loc[df["Temperature"].idxmax()]
lowest_temp_row = df.loc[df["Temperature"].idxmin()]

print("\nRow with Highest Temperature:")
print(highest_temp_row)

print("\nRow with Lowest Temperature:")
print(lowest_temp_row)

feature_std = df[corr_features].std()
print("\nStandard Deviation of Weather Features:\n", feature_std)

plt.figure(figsize=(10, 5))
plt.plot(df.index, df["Temperature"], marker='o')
plt.title("Temperature Trend")
plt.xlabel("Index")
plt.ylabel("Temperature")
plt.grid(True)
plt.tight_layout()
plt.show()

# ===================================================
# EXPERIMENT 10: WEATHER ANALYSIS
# ===================================================
print("\n\n===== EXPERIMENT 10: WEATHER ANALYSIS =====")

print("Average Wind Speed:", df["Wind_Speed"].mean())
print("Average Humidity:", df["Humidity"].mean())
print("Average Pressure:", df["Pressure"].mean())
print("Total Rain Value:", df["Rain_Binary"].sum())

plt.figure(figsize=(8, 5))
sns.scatterplot(x="Humidity", y="Temperature", data=df, alpha=0.7)
plt.title("Humidity vs Temperature")
plt.xlabel("Humidity")
plt.ylabel("Temperature")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x="Pressure", y="Temperature", data=df, alpha=0.7)
plt.title("Pressure vs Temperature")
plt.xlabel("Pressure")
plt.ylabel("Temperature")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n===== ALL 10 EXPERIMENT CONCEPTS COMPLETED USING YOUR WEATHER DATASET =====")