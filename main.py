import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Black Friday Sale Analysis",
    layout="wide"
)

# =========================
# APP TITLE
# =========================
st.title("Black Friday Sale Analysis Dashboard")

# =========================
# DEFAULT DATA URL
# =========================
data_url = "https://raw.githubusercontent.com/Tiffany-14/Black_Friday_Sale/main/Black_Friday_Sale.csv"

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data(url):
    return pd.read_csv(url)

df = load_data(data_url)

# =========================
# DATA PREVIEW
# =========================
st.subheader("📄 Dataset Overview")

col1, col2 = st.columns(2)
with col1:
    st.write("Shape:", df.shape)
with col2:
    st.write("Missing Values:")
    st.write(df.isnull().sum())

st.dataframe(df.head())

# =========================
# DATA CLEANING
# =========================
st.subheader("🧹 Data Cleaning")

df['Product_Category_2'] = df['Product_Category_2'].fillna(0)
df['Product_Category_3'] = df['Product_Category_3'].fillna(0)

st.success("✔ Missing values handled successfully")

# =========================
# EDA
# =========================
st.subheader("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.histplot(df['Purchase'], bins=30, kde=True, ax=ax)
    ax.set_title("Distribution of Purchase Amount")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.boxplot(x='Gender', y='Purchase', data=df, ax=ax)
    ax.set_title("Purchase by Gender")
    st.pyplot(fig)

# =========================
# KMEANS CLUSTERING
# =========================
st.subheader("🤖 Customer Segmentation (KMeans)")

k = st.slider("Select number of clusters (K)", 2, 10, 4)

cluster_df = df[['Age', 'Purchase']].copy()
cluster_df['Age'] = cluster_df['Age'].astype('category').cat.codes

scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_df)

kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

fig, ax = plt.subplots()
sns.scatterplot(
    x=df['Age'],
    y=df['Purchase'],
    hue=df['Cluster'],
    palette='tab10',
    ax=ax
)
ax.set_title("Customer Clusters")
st.pyplot(fig)

# =========================
# LINEAR REGRESSION
# =========================
st.subheader("📈 Purchase Prediction (Linear Regression)")

model_df = df.copy()

model_df['Age'] = model_df['Age'].astype('category').cat.codes
model_df['City_Category'] = model_df['City_Category'].astype('category').cat.codes

model_df = pd.get_dummies(
    model_df,
    columns=['Gender', 'Occupation'],
    drop_first=True
)

X = model_df.drop('Purchase', axis=1)
y = model_df['Purchase']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
col2.metric("MSE", f"{mean_squared_error(y_test, y_pred):.2f}")
col3.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")

st.success("✔ Model trained and evaluated successfully")
