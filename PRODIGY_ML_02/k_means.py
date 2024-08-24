import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import streamlit as st

# Load dataset

def load_data():
    df = pd.read_csv('C:/Users/HP/Documents/PRODIGY_ML/PRODIGY_ML_02/customer_data.csv')

    # Preprocess the data
    def percentage_to_float(pct_str):
        return float(pct_str.strip('%')) / 100

    df['Electronics Purchases'] = df['Electronics Purchases'].apply(percentage_to_float)
    df['Clothing Purchases'] = df['Clothing Purchases'].apply(percentage_to_float)
    df['Grocery Purchases'] = df['Grocery Purchases'].apply(percentage_to_float)
    df['Discount Sensitivity'] = df['Discount Sensitivity'].apply(percentage_to_float)

    le = LabelEncoder()
    df['Frequency of Purchases'] = df['Frequency of Purchases'].apply(lambda x: int(x.split()[0]))
    df['Seasonal Spending Pattern'] = le.fit_transform(df['Seasonal Spending Pattern'])
    df['Location'] = le.fit_transform(df['Location'])

    return df

df = load_data()

# Selected features for clustering
features = ['Total Purchase Amount', 'Number of Transactions', 'Frequency of Purchases',
            'Recency of Last Purchase', 'Electronics Purchases', 'Clothing Purchases', 'Grocery Purchases']

# Handle 'Recency of Last Purchase' by converting days to numeric
df['Recency of Last Purchase'] = df['Recency of Last Purchase'].apply(lambda x: int(x.split()[0]))

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Fit the K-means model
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Map clusters to customer types
cluster_labels = {0: 'Regular', 1: 'Occasional', 2: 'Rare'}
df['Customer Type'] = df['Cluster'].map(cluster_labels)

# Streamlit UI
st.title('Customer Clustering App')

st.sidebar.header('Input Features')

def user_input_features():
    total_purchase_amount = st.sidebar.number_input('Total Purchase Amount', min_value=0)
    number_of_transactions = st.sidebar.number_input('Number of Transactions', min_value=0)
    frequency_of_purchases = st.sidebar.selectbox('Frequency of Purchases', options=[1, 2, 3])
    recency_of_last_purchase = st.sidebar.number_input('Recency of Last Purchase (days)', min_value=0)
    electronics_purchases = st.sidebar.slider('Electronics Purchases (%)', 0, 100, 0) / 100
    clothing_purchases = st.sidebar.slider('Clothing Purchases (%)', 0, 100, 0) / 100
    grocery_purchases = st.sidebar.slider('Grocery Purchases (%)', 0, 100, 0) / 100

    data = {
        'Total Purchase Amount': total_purchase_amount,
        'Number of Transactions': number_of_transactions,
        'Frequency of Purchases': frequency_of_purchases,
        'Recency of Last Purchase': recency_of_last_purchase,
        'Electronics Purchases': electronics_purchases,
        'Clothing Purchases': clothing_purchases,
        'Grocery Purchases': grocery_purchases
    }
    return pd.DataFrame(data, index=[0])

user_input = user_input_features()

# Preprocess user input
user_input_scaled = scaler.transform(user_input[features])

# Predict the cluster
user_cluster = kmeans.predict(user_input_scaled)
user_type = cluster_labels[user_cluster[0]]

st.write(f'**Customer Type:** {user_type}')
