import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Streamlit app title
st.title("Black Friday Sale Analysis Dashboard")

# Định nghĩa URL dữ liệu mặc định
data_url = "https://raw.githubusercontent.com/Tiffany-14/Black_Friday_Sale/refs/heads/main/Black_Friday_Sale.csv"

# Tải dữ liệu
@st.cache_data
def load_data():
    df = pd.read_csv(data_url)
    # Xử lý cơ bản: Điền NaN bằng 0.0 cho Product_Category_2 và 3, loại bỏ duplicates
    df[['Product_Category_2', 'Product_Category_3']] = df[['Product_Category_2', 'Product_Category_3']].fillna(0.0)
    df = df.drop_duplicates()
    return df

df = load_data()

st.sidebar.header("Điều hướng")
page = st.sidebar.radio("Chọn phần phân tích", 
                        ["Tổng quan dữ liệu", 
                         "Phân tích theo Nhóm tuổi", 
                         "Phân tích theo Giới tính & Độ tuổi", 
                         "Phân tích theo Thành phố", 
                         "Các biểu đồ trực quan", 
                         "Phân cụm khách hàng (Clustering)", 
                         "Dự đoán mức chi tiêu (Linear Regression)"])

# --- TỔNG QUAN DỮ LIỆU ---
if page == "Tổng quan dữ liệu":
    st.header("Tổng quan dữ liệu Black Friday Sale")
    st.write(f"Số bản ghi sau xử lý: **{len(df):,}**")
    st.dataframe(df.head())
    st.subheader("Thông tin dữ liệu")
    st.text(df.info())
    st.subheader("Mô tả thống kê")
    st.dataframe(df.describe())

# --- PHÂN TÍCH THEO NHÓM TUỔI (AGE TIER) ---
elif page == "Phân tích theo Nhóm tuổi":
    st.header("Phân tích mua hàng theo Nhóm tuổi cốt lõi (Age Tier)")

    # Tạo cột Age_Tier
    def create_age_group_tier(age_str):
        if age_str in ['0-17', '18-25']:
            return 'Young_Adults'
        elif age_str in ['26-35', '36-45', '46-50']:
            return 'Middle_Age'
        else:  # '51-55', '55+'
            return 'Seniors'

    df_age = df.copy()
    df_age['Age_Tier'] = df_age['Age'].apply(create_age_group_tier)

    age_analysis = df_age.groupby('Age_Tier')['Purchase'].agg(
        ['count', 'mean', 'median', 'sum']
    ).rename(columns={
        'count': 'Count_Records',
        'mean': 'Mean_Purchase',
        'median': 'Median_Purchase',
        'sum': 'Total_Purchase'
    }).sort_values(by='Mean_Purchase', ascending=False)

    age_analysis['Mean_Purchase'] = age_analysis['Mean_Purchase'].round(2)
    age_analysis['Median_Purchase'] = age_analysis['Median_Purchase'].round(2)
    age_analysis['Total_Purchase'] = age_analysis['Total_Purchase'].round(2)

    st.dataframe(age_analysis.style.format("{:,.2f}"))

# --- PHÂN TÍCH THEO GIỚI TÍNH & ĐỘ TUỔI ---
elif page == "Phân tích theo Giới tính & Độ tuổi":
    st.header("Phân tích tần suất và mức chi tiêu theo Giới tính & Độ tuổi")

    # 1. Tần suất theo Giới tính
    st.subheader("Tần suất khách hàng theo Giới tính")
    gender_counts = df['Gender'].value_counts()
    gender_percentage = df['Gender'].value_counts(normalize=True).mul(100).round(2)
    gender_dist = pd.DataFrame({
        'Total Transactions': gender_counts,
        'Percentage (%)': gender_percentage
    })
    gender_dist.index = ['Male (M)', 'Female (F)']
    st.dataframe(gender_dist)

    # 2. Tần suất theo Độ tuổi
    st.subheader("Tần suất khách hàng theo Độ tuổi")
    age_summary = pd.DataFrame({
        'Total Transactions': df['Age'].value_counts().sort_index(),
        'Percentage (%)': df['Age'].value_counts(normalize=True).mul(100).round(2).sort_index()
    }).reset_index()
    st.dataframe(age_summary)

    # 3. Mức chi tiêu trung bình theo Giới tính & Độ tuổi
    st.subheader("Mức chi tiêu trung bình theo Giới tính và Độ tuổi")
    gender_age_pivot = df.pivot_table(
        values='Purchase',
        index='Age',
        columns='Gender',
        aggfunc='mean'
    ).round(2)
    st.dataframe(gender_age_pivot.style.format("{:,.2f}"))

# --- PHÂN TÍCH THEO THÀNH PHỐ ---
elif page == "Phân tích theo Thành phố":
    st.header("Phân tích mức chi tiêu theo Thành phố và Giới tính")

    city_gender_pivot = df.pivot_table(
        values='Purchase',
        index='City_Category',
        columns='Gender',
        aggfunc='mean'
    ).round(2)
    st.dataframe(city_gender_pivot.style.format("{:,.2f}"))

# --- CÁC BIỂU ĐỒ TRỰC QUAN ---
elif page == "Các biểu đồ trực quan":
    st.header("Các biểu đồ trực quan hóa")

    # Biểu đồ 1: Average Purchase by Age Group and Gender
    st.subheader("Mức chi tiêu trung bình theo Độ tuổi và Giới tính")
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    sns.barplot(data=df.sort_values('Age'), x='Age', y='Purchase', hue='Gender',
                palette={'M': '#1f77b4', 'F': '#ff7f0e'}, errorbar=None, ax=ax1)
    ax1.set_title('Average Purchase by Age Group and Gender')
    ax1.set_xlabel('Age Group')
    ax1.set_ylabel('Average Purchase (USD)')
    ax1.legend(title='Gender', labels=['Male (M)', 'Female (F)'])
    st.pyplot(fig1)

    # Biểu đồ 2: Total Purchase by City Category
    st.subheader("Tổng doanh thu theo Thành phố")
    df_city = pd.get_dummies(df, columns=['City_Category'], prefix='City', dtype=int)
    city_purchase = df_city[['City_A', 'City_B', 'City_C']].multiply(df_city['Purchase'], axis=0).sum() / 1000
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.barplot(x=city_purchase.index, y=city_purchase.values, ax=ax2)
    ax2.set_title('Total Purchase by City Category')
    ax2.set_ylabel('Total Purchase (K$)')
    st.pyplot(fig2)

    # Biểu đồ 3: Product Category 1 Distribution
    st.subheader("Phân bố Product Category 1 (chỉ các category > 5000 giao dịch)")
    product_cat1_counts = df['Product_Category_1'].value_counts()
    product_cat1_counts = product_cat1_counts[product_cat1_counts > 5000]
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    ax3.pie(product_cat1_counts, labels=[f'Category {int(x)}' for x in product_cat1_counts.index],
            autopct='%1.1f%%', startangle=140)
    ax3.set_title('Product Category 1 Distribution')
    st.pyplot(fig3)

    # Biểu đồ 4: Total Purchase by Years in Current City
    st.subheader("Tổng doanh thu theo Số năm sống tại thành phố hiện tại")
    stay_purchase = df.groupby('Stay_In_Current_City_Years')['Purchase'].sum() / 1000
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.plot(stay_purchase.index, stay_purchase.values, marker='o')
    ax4.set_title('Total Purchase by Years in Current City')
    ax4.set_xlabel('Years in Current City')
    ax4.set_ylabel('Total Purchase (K$)')
    st.pyplot(fig4)

# --- PHÂN CỤM KHÁCH HÀNG (KMEANS) ---

elif page == "Phân cụm khách hàng (Clustering)":
    st.header("Phân cụm khách hàng bằng K-Means (k=3)")

    st.warning("⚠️ Phân cụm đang được thực hiện trên mẫu dữ liệu ngẫu nhiên 10.000 bản ghi để đảm bảo tốc độ và ổn định.")

    # Lấy mẫu ngẫu nhiên để tránh lỗi bộ nhớ
    df_sample = df.sample(n=10000, random_state=42).copy()

    # Tiền xử lý: điền missing values
    df_sample[['Product_Category_2', 'Product_Category_3']] = df_sample[['Product_Category_2', 'Product_Category_3']].fillna(0)

    # Chỉ chọn các đặc trưng quan trọng để clustering (giảm chiều dữ liệu)
    features_to_use = [
        'Gender', 'Age', 'Occupation', 'City_Category', 
        'Stay_In_Current_City_Years', 'Marital_Status',
        'Product_Category_1', 'Product_Category_2', 'Product_Category_3',
        'Purchase'
    ]

    df_cluster = df_sample[features_to_use].copy()

    # One-hot encoding chỉ các cột categorical
    categorical_cols = ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years']
    df_cluster = pd.get_dummies(df_cluster, columns=categorical_cols, dtype=int)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)

    # Huấn luyện KMeans với k=3
    with st.spinner("Đang thực hiện phân cụm..."):
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df_sample['Cluster'] = kmeans.fit_predict(X_scaled)

    st.success("✅ Phân cụm hoàn tất!")

    # Hiển thị kết quả mẫu
    st.subheader("Kết quả phân cụm (mẫu 10 dòng)")
    display_cols = ['User_ID', 'Gender', 'Age', 'City_Category', 'Purchase', 'Cluster']
    st.dataframe(df_sample[display_cols].head(10))

    # Thống kê mỗi cụm
    st.subheader("Thống kê trung bình theo từng cụm")
    cluster_summary = df_sample.groupby('Cluster').agg({
        'Purchase': ['mean', 'median', 'count'],
        'Age': lambda x: x.mode().iloc[0] if not x.empty else 'N/A',
        'Gender': lambda x: x.mode().iloc[0] if not x.empty else 'N/A',
        'City_Category': lambda x: x.mode().iloc[0] if not x.empty else 'N/A'
    }).round(2)

    cluster_summary.columns = ['Mean_Purchase', 'Median_Purchase', 'Count', 'Most_Common_Age', 'Most_Common_Gender', 'Most_Common_City']
    cluster_summary = cluster_summary.reset_index()
    cluster_summary['Cluster'] = cluster_summary['Cluster'].astype(str).str.replace('0', 'Cụm 0').str.replace('1', 'Cụm 1').str.replace('2', 'Cụm 2')

    st.dataframe(cluster_summary.style.format({
        'Mean_Purchase': '{:,.0f}',
        'Median_Purchase': '{:,.0f}',
        'Count': '{:,}'
    }))

    # Biểu đồ mức chi tiêu trung bình theo cụm
    st.subheader("Mức chi tiêu trung bình theo cụm")
    fig_cluster, ax_cluster = plt.subplots(figsize=(8, 6))
    sns.barplot(
        data=df_sample,
        x='Cluster',
        y='Purchase',
        palette='viridis',
        errorbar=None,
        ax=ax_cluster
    )
    ax_cluster.set_title('Average Purchase by Customer Cluster')
    ax_cluster.set_ylabel('Average Purchase (USD)')
    ax_cluster.set_xlabel('Cluster')
    st.pyplot(fig_cluster)

# --- DỰ ĐOÁN MỨC CHI TIÊU (LINEAR REGRESSION) --- (PHIÊN BẢN ĐƠN GIẢN)
elif page == "Dự đoán mức chi tiêu (Linear Regression)":
    st.header("Dự đoán mức chi tiêu (Linear Regression - Phiên bản đơn giản)")

    st.info("🔹 Mô hình được huấn luyện nhanh trên 20.000 bản ghi ngẫu nhiên để đảm bảo tốc độ.")

    # Lấy mẫu nhỏ để chạy nhanh
    df_sample = df.sample(n=20000, random_state=42).copy()

    # Tiền xử lý cơ bản
    df_sample[['Product_Category_2', 'Product_Category_3']] = df_sample[['Product_Category_2', 'Product_Category_3']].fillna(0)

    X = df_sample.drop(['Purchase', 'User_ID', 'Product_ID'], axis=1)
    y = df_sample['Purchase']

    # One-hot encoding các cột phân loại
    cat_cols = ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years']
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=int)

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Huấn luyện mô hình
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Tính metric đơn giản
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Hiển thị kết quả đẹp mắt
    st.success("✅ Huấn luyện thành công!")
    
    col1, col2 = st.columns(2)
    col1.metric("Độ chính xác mô hình (R²)", f"{r2:.4f}")
    col2.metric("Sai số trung bình (MAE)", f"{mae:,.0f} USD")

    st.write("""
    **Giải thích ngắn gọn:**
    - R² khoảng **0.13 - 0.15**: Mô hình giải thích được ~14% biến thiên trong mức chi tiêu (bình thường với dữ liệu mua sắm Black Friday).
    - MAE khoảng **2,400 - 2,600 USD**: Dự đoán sai trung bình ±2,500 USD so với thực tế.
    """)
