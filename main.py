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

# --- DỰ ĐOÁN MỨC CHI TIÊU (LINEAR REGRESSION)
elif page == "Dự đoán mức chi tiêu (Linear Regression)":
    st.header("Mô hình dự đoán mức chi tiêu bằng Linear Regression")

    st.warning("⚠️ Để đảm bảo tốc độ và ổn định, mô hình được huấn luyện trên mẫu ngẫu nhiên 20.000 bản ghi.")

    # Lấy mẫu dữ liệu để tránh lỗi bộ nhớ
    df_model = df.sample(n=20000, random_state=42).copy()

    # Tiền xử lý: Điền NaN (đã làm ở load_data, nhưng đảm bảo lại)
    df_model[['Product_Category_2', 'Product_Category_3']] = df_model[['Product_Category_2', 'Product_Category_3']].fillna(0)

    # Chuẩn bị features và target
    X = df_model.drop(['Purchase', 'User_ID', 'Product_ID'], axis=1)
    y = df_model['Purchase']

    # Chỉ one-hot encoding các cột categorical có ít giá trị unique (an toàn cho bộ nhớ)
    categorical_cols = ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']
    
    # Các cột numerical giữ nguyên
    numerical_cols = ['Occupation', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3']

    # One-hot encoding
    X_encoded = pd.get_dummies(X[categorical_cols], columns=categorical_cols, drop_first=True, dtype=int)
    
    # Kết hợp lại với numerical columns
    X_final = pd.concat([X[numerical_cols], X_encoded], axis=1)

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

    # Huấn luyện mô hình
    with st.spinner("Đang huấn luyện mô hình Linear Regression..."):
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    st.success("✅ Huấn luyện mô hình hoàn tất!")

    # Đánh giá
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.subheader("Kết quả đánh giá mô hình")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Squared Error (MSE)", f"{mse:,.0f}")
    col2.metric("R-squared (R²)", f"{r2:.4f}")
    col3.metric("Mean Absolute Error (MAE)", f"{mae:,.0f}")

    st.info("""
    **Giải thích nhanh:**
    - R² ≈ 0.13–0.15 là bình thường với dữ liệu Black Friday (Linear Regression đơn giản, không phải mô hình phức tạp).
    - MAE ≈ 3,500–4,000 USD nghĩa là dự đoán sai lệch trung bình khoảng ±3.500–4.000 USD so với thực tế.
    """)

    # Biểu đồ so sánh thực tế vs dự đoán (mẫu nhỏ)
    st.subheader("So sánh Giá trị thực tế vs Dự đoán (mẫu 100 điểm)")
    compare_df = pd.DataFrame({
        'Actual': y_test.values[:100],
        'Predicted': y_pred[:100]
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(range(100), compare_df['Actual'], color='blue', label='Actual', alpha=0.7)
    ax.scatter(range(100), compare_df['Predicted'], color='red', label='Predicted', alpha=0.7)
    ax.plot(range(100), compare_df['Predicted'], color='red', alpha=0.5, linestyle='--')
    ax.set_title('Actual vs Predicted Purchase (Sample 100)')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Purchase Amount (USD)')
    ax.legend()
    st.pyplot(fig)
