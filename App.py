import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Supervised Learning – Regression Lab",
    layout="wide"
)

# -------------------------------------------------
# LOAD CSS
# -------------------------------------------------
def load_css(file):
    try:
        with open(file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass

load_css("style.css")

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.markdown("""
<div class="card">
<h1>Supervised Learning – Regression Analysis</h1>
<p>Understand → Analyze → Clean → Model → Visualize → Predict</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])

if uploaded_file is None:
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except:
    st.error("❌ Failed to read CSV file.")
    st.stop()

if df.shape[0] == 0:
    st.error("❌ Uploaded dataset is empty.")
    st.stop()

# =================================================
# BEFORE CLEANING ANALYSIS
# =================================================
st.subheader("🔍 Dataset Preview (Before Cleaning)")
st.dataframe(df.head())

st.subheader("🧪 Dataset Quality Analysis (Before Cleaning)")

missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
st.dataframe(pd.DataFrame({
    "Column": df.columns,
    "Missing Count": missing.values,
    "Missing %": missing_percent.values
}))

outlier_info = {}
for col in df.select_dtypes(include=np.number).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outlier_info[col] = ((df[col] < Q1 - 1.5 * IQR) |
                         (df[col] > Q3 + 1.5 * IQR)).sum()

st.dataframe(pd.DataFrame({
    "Column": outlier_info.keys(),
    "Outlier Count": outlier_info.values()
}))

st.markdown("""
<div style="background:#0f172a;color:#e5e7eb;padding:18px;border-radius:12px;">
<h3>🔊 Noise Possibility Analysis</h3>
<ul>
<li>Measurement or data entry errors</li>
<li>Missing or unobserved variables</li>
<li>High variance in target values</li>
<li>Weak or inconsistent relationships</li>
</ul>
<p>Noise is later inferred from model errors and residuals.</p>
</div>
""", unsafe_allow_html=True)

# =================================================
# DATA CLEANING
# =================================================
st.subheader("🧹 Data Cleaning")

for col in df.columns:
    if df[col].dtype == "object":
        if df[col].isnull().all():
            df[col] = df[col].fillna("Unknown")
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

cat_cols = df.select_dtypes(include=["object", "category"]).columns
for col in cat_cols:
    df[col] = df[col].astype(str)
    df[col] = LabelEncoder().fit_transform(df[col])

num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_pct = (outlier_count / len(df)) * 100

    if outlier_pct < 30:
        df = df[~((df[col] < lower) | (df[col] > upper))]
    else:
        median = df[col].median()
        df[col] = np.where(
            (df[col] < lower) | (df[col] > upper),
            median,
            df[col]
        )

if df.shape[0] < 10:
    st.error("❌ Dataset too small after cleaning.")
    st.stop()

st.success("✅ Data cleaned successfully!")
st.subheader("✅ Dataset Preview (After Cleaning)")
st.dataframe(df.head())

# =================================================
# TARGET SELECTION
# =================================================
st.subheader("🎯 Target Variable")
target = st.selectbox("Select Dependent Variable (Y)", df.columns)

# =================================================
# MODEL SELECTION (NO DEFAULT)
# =================================================
st.subheader("🤖 Regression Model Selection")

model_type = st.selectbox(
    "Select Regression Model",
    [
        "Select a model",
        "Simple Linear Regression",
        "Multiple Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
        "ElasticNet Regression"
    ],
    index=0
)

if model_type == "Select a model":
    st.warning("Please select a regression model to proceed.")
    st.stop()

# =================================================
# INDEPENDENT VARIABLES
# =================================================
st.subheader("📌 Independent Variables")

available_features = [c for c in df.columns if c != target]

if model_type == "Simple Linear Regression":
    features = [st.selectbox(
        "Select ONLY ONE Independent Variable (X)",
        available_features
    )]
else:
    features = st.multiselect(
        "Select Independent Variables (X)",
        available_features,
        default=available_features[:min(2, len(available_features))]
    )

if not features:
    st.stop()

# =================================================
# TRAINING
# =================================================
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model_map = {
    "Simple Linear Regression": LinearRegression(),
    "Multiple Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "ElasticNet Regression": ElasticNet(alpha=0.01, l1_ratio=0.5)
}

model = model_map[model_type]
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

# =================================================
# PERFORMANCE
# =================================================
st.subheader("📊 Model Performance")

c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
c3.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")

# =================================================
# REGRESSION EQUATION
# =================================================
st.subheader("📐 Regression Equation")

equation = f"y = {model.intercept_:.3f}"
for coef, feat in zip(model.coef_, features):
    equation += f" + ({coef:.3f} × {feat})"

st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #1e293b, #0f172a);
    color: #e5e7eb;
    padding: 18px;
    border-radius: 14px;
    font-size: 18px;
    font-weight: 600;
    border-left: 5px solid #38bdf8;
">
{equation}
</div>
""", unsafe_allow_html=True)

# =================================================
# INTERPRETATION
# =================================================
st.subheader("📘 Model Interpretation")

st.markdown(
    f"📍 **Intercept (b₀): {model.intercept_:.3f}** → "
    f"When all independent variables are zero, "
    f"the predicted **{target}** is **{model.intercept_:.3f}**."
)

for coef, feat in zip(model.coef_, features):
    st.markdown(
        f"📌 **Slope for {feat} (b): {coef:.3f}** → "
        f"For every 1 unit increase in **{feat}**, "
        f"the target **{target}** changes by **{coef:.3f} units**."
    )

# =================================================
# VISUALIZATION
# =================================================
st.subheader("📈 Model Visualization")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.6)
ax.plot([y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red")
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
st.pyplot(fig)

# =================================================
# PREDICTION (BIG DISPLAY)
# =================================================
st.subheader("✨ Make a Prediction")

input_data = {}
for col in features:
    min_val = float(df[col].min())
    max_val = float(df[col].max())

    if min_val == max_val:
        input_data[col] = min_val
        st.info(f"{col} is constant ({min_val})")
    else:
        input_data[col] = st.number_input(col, min_val, max_val)

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)

st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: #f8fafc;
    padding: 30px;
    border-radius: 20px;
    margin-top: 25px;
    text-align: center;
    box-shadow: 0px 15px 35px rgba(0,0,0,0.35);
    border-left: 6px solid #38bdf8;
">
    <h2>🔮 Prediction Result</h2>
    <h1 style="font-size:48px;color:#22d3ee;margin:10px 0;">
        {prediction[0]:.2f}
    </h1>
    <p style="font-size:18px;">
        Predicted <b>{target}</b> based on selected input values
    </p>
</div>
""", unsafe_allow_html=True)
