import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from skimage.metrics import structural_similarity as ssim
import cv2


# Set wide layout
st.set_page_config(layout="wide")

# ---- Sidebar ----
st.sidebar.title("🔋 Ev Energy Dashboard")
page = st.sidebar.radio("Go to", [
    "Home", 
    "Driver Insights", 
    "Forecasting", 
    "Clustering", 
    "Map view", 
    "Cost Analysis", 
    "Simulation"
])

# ---- Load data ----
@st.cache_data
def load_energy_data():
    df = pd.read_csv("data/ev_energy_data.csv", parse_dates=["timestamp"])
    return df

@st.cache_data
def load_driver_data():
    df = pd.read_csv("data/driver_profiles.csv")
    return df

energy_df = load_energy_data()
driver_df = load_driver_data()

# ---- Home Page ----
if page == "Home":
    st.title("🚗 Ev Energy Forecasting & Driver Behavior Dashboard")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Overview of Ev Energy Usage")
        st.dataframe(energy_df.head())

    with col2:
        st.metric("Total Drivers", len(driver_df))
        st.metric("Unique vehicle Types", energy_df['vehicle_type'].nunique())
        st.metric("Avg Energy (kWh)", round(energy_df['energy_kwh'].mean(), 2))

    with st.expander("🔍 See Sample Driver Profiles"):
        st.dataframe(driver_df.head())

    st.success("Use the sidebar to explore driver behavior, forecasting, clustering, and more.")

# Driver Insights Page
if page == "Driver Insights":
    st.title("🧑‍💼 Driver Insights")

    # Merge energy and driver data
    # Ensure 'driver_id' in both DataFrames is string (object) type before merging
    energy_df['driver_id'] = energy_df['driver_id'].astype(str)
    driver_df['driver_id'] = driver_df['driver_id'].astype(str)

    # Now merge safely
    df = pd.merge(energy_df, driver_df, on='driver_id', how='left')

    # Sidebar filters
    st.sidebar.subheader("Filter Options")
    selected_branch = st.sidebar.selectbox("Select Branch", options=["All"] + sorted(df['branch'].dropna().unique().tolist()))
    selected_shift = st.sidebar.selectbox("Select Shift", options=["All"] + sorted(df['shift'].dropna().unique().tolist()))
    selected_style = st.sidebar.selectbox("Select Driving Style", options=["All"] + sorted(df['driving_style'].dropna().unique().tolist()))

    # Apply filters
    filtered_df = df.copy()
    if selected_branch != "All":
        filtered_df = filtered_df[filtered_df['branch'] == selected_branch]
    if selected_shift != "All":
        filtered_df = filtered_df[filtered_df['shift'] == selected_shift]
    if selected_style != "All":
        filtered_df = filtered_df[filtered_df['driving_style'] == selected_style]

    # Display filtered data
    st.subheader("Filtered Driver Data")
    st.dataframe(filtered_df.head())

    # Summary statistics
    st.subheader("Summary Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_energy = filtered_df['energy_kwh'].mean()
        st.metric("Average Energy Consumption (kWh)", f"{avg_energy:.2f}")
    with col2:
        avg_charging_duration = filtered_df['charging_duration_minutes'].mean()
        st.metric("Average Charging Duration (minutes)", f"{avg_charging_duration:.2f}")
    with col3:
        avg_risk_score = filtered_df['sensor_risk_score'].mean()
        st.metric("Average Sensor Risk Score", f"{avg_risk_score:.2f}")

    # Risk score distribution
    st.subheader("Sensor Risk Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df['sensor_risk_score'], bins=20, kde=True, ax=ax)
    ax.set_xlabel("Sensor Risk Score")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Energy consumption by shift
    st.subheader("Average Energy Consumption by Shift")
    energy_by_shift = filtered_df.groupby('shift')['energy_kwh'].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(data=energy_by_shift, x='shift', y='energy_kwh', ax=ax)
    ax.set_xlabel("Shift")
    ax.set_ylabel("Average Energy Consumption (kWh)")
    st.pyplot(fig)

    # Energy consumption by driving style
    st.subheader("Average Energy Consumption by Driving Style")
    energy_by_style = filtered_df.groupby('driving_style')['energy_kwh'].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(data=energy_by_style, x='driving_style', y='energy_kwh', ax=ax)
    ax.set_xlabel("Driving Style")
    ax.set_ylabel("Average Energy Consumption (kWh)")
    st.pyplot(fig)


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from skimage.metrics import structural_similarity as ssim
import cv2

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    df = df.sort_values("Date")
    return df

def plot_series(df, title):
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["Sales"], label="Sales")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    st.pyplot(fig)

def forecast_arima(df):
    model = ARIMA(df["Sales"], order=(5,1,0))
    fitted = model.fit()
    forecast = fitted.forecast(steps=12)
    return forecast

def forecast_lstm(df):
    scaler = MinMaxScaler()
    sales_scaled = scaler.fit_transform(df["Sales"].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(10, len(sales_scaled)):
        X.append(sales_scaled[i-10:i])
        y.append(sales_scaled[i])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, verbose=0)

    # Forecast 12 steps
    predictions = []
    last_sequence = sales_scaled[-10:]
    current_input = last_sequence.reshape(1, 10, 1)
    for _ in range(12):
        pred = model.predict(current_input)[0][0]
        predictions.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions

def generate_ssim_heatmap(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    score, diff = ssim(img1, img2, full=True)
    heatmap = (diff * 255).astype("uint8")
    return heatmap, score

# Forecasting Page
if page == "Forecasting":
    st.header("📈 Forecasting with ARIMA and LSTM")
    uploaded_file = st.file_uploader("Upload time series CSv (with 'Date' and 'Sales' columns)", type=["csv"])

    if uploaded_file:
        df = load_data(uploaded_file)
    else:
        # Use synthetic data if no upload
        date_range = pd.date_range(start="2015-01-01", periods=120, freq="M")
        sales = np.cumsum(np.random.normal(loc=100, scale=10, size=120))
        df = pd.DataFrame({"Date": date_range, "Sales": sales})

    st.subheader("📊 Original Time Series Data")
    st.dataframe(df.head())
    plot_series(df, "Original Sales Data")

    st.subheader("🔮 Forecast with ARIMA")
    arima_forecast = forecast_arima(df)
    arima_df = pd.DataFrame({
        "Date": pd.date_range(df["Date"].iloc[-1] + pd.offsets.MonthEnd(1), periods=12, freq="M"),
        "Forecast": arima_forecast
    })
    st.line_chart(arima_df.set_index("Date"))

    st.subheader("🧠 Forecast with LSTM")
    lstm_forecast = forecast_lstm(df)
    lstm_df = pd.DataFrame({
        "Date": pd.date_range(df["Date"].iloc[-1] + pd.offsets.MonthEnd(1), periods=12, freq="M"),
        "Forecast": lstm_forecast
    })
    st.line_chart(lstm_df.set_index("Date"))

    # Plot actual + forecasts side-by-side for SSIM
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["Sales"], label="Actual")
    ax.plot(arima_df["Date"], arima_df["Forecast"], label="ARIMA Forecast")
    ax.plot(lstm_df["Date"], lstm_df["Forecast"], label="LSTM Forecast")
    ax.legend()
    st.pyplot(fig)

    # Save plots as images for SSIM comparison
    fig1, ax1 = plt.subplots()
    ax1.plot(df["Sales"])
    ax1.set_title("Original")
    fig1.canvas.draw()
    img1 = np.frombuffer(fig1.canvas.tostring_rgb(), dtype=np.uint8)
    img1 = img1.reshape(fig1.canvas.get_width_height()[::-1] + (3,))

    fig2, ax2 = plt.subplots()
    ax2.plot(lstm_df["Forecast"])
    ax2.set_title("LSTM Forecast")
    fig2.canvas.draw()
    img2 = np.frombuffer(fig2.canvas.tostring_rgb(), dtype=np.uint8)
    img2 = img2.reshape(fig2.canvas.get_width_height()[::-1] + (3,))

    heatmap, score = generate_ssim_heatmap(img1, img2)
    st.subheader(f"🧪 SSIM Heatmap (Similarity Score: {score:.2f})")
    st.image(heatmap, caption="SSIM Heatmap: Original vs LSTM Forecast")
if page == "Clustering":
    st.title("👥 Driver Clustering by Profile")

    df = driver_df.copy()
    st.write("Driver profile columns:", df.columns.tolist())  # Debug output

    st.write("Driver profile columns:", df.columns.tolist())


    # Adjust these based on actual column names
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=['age', 'vehicleHours'])

    features = ['age', 'vehicleHours']
    X = df_clean[features]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    from sklearn.cluster import KMeans
    k = st.slider("Select number of clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    df_clean['cluster'] = kmeans.fit_predict(X_scaled)

    st.subheader("📊 Cluster Summary")
    st.dataframe(df_clean[['driver_id', 'age', 'vehicleHours', 'cluster']].head())

    st.subheader("📈 Cluster Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df_clean, x='cluster', palette='viridis')
    st.pyplot(fig)

    st.subheader("🧬 Cluster Scatter Plot (age vs Usage)")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df_clean, x='age', y='vehicleHours', hue='cluster', palette='tab10')
    ax2.set_xlabel("Driver age")
    ax2.set_ylabel("vehicle Usage Hours")
    st.pyplot(fig2)

    st.info("Driver clusters help understand behavior patterns among drivers.")
