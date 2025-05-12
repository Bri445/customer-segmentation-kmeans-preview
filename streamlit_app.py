import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🧠 Customer Segmentation using K-Means")

# Upload CSV
uploaded_file = st.file_uploader("Upload Mall_Customers.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📊 Dataset Preview", df.head())

    # Select columns for clustering
    st.write("### 🎯 Select Features for Clustering")
    options = st.multiselect("Select two numerical features", df.select_dtypes(include='number').columns.tolist(), default=["Annual Income (k$)", "Spending Score (1-100)"])
    
    if len(options) == 2:
        X = df[options]

        # Elbow Method Plot
        sse = []
        for k in range(1, 11):
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(X)
            sse.append(km.inertia_)
        plt.figure()
        plt.plot(range(1, 11), sse, marker='o')
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        plt.title("Elbow Method")
        st.pyplot(plt)

        # Choose number of clusters
        k = st.slider("Select number of clusters", 2, 10, 5)
        model = KMeans(n_clusters=k, random_state=42)
        df["Cluster"] = model.fit_predict(X)

        # Show results
        st.write("### 🧩 Clustered Data", df.head())

        plt.figure()
        sns.scatterplot(x=X[options[0]], y=X[options[1]], hue=df["Cluster"], palette="Set2")
        plt.title("Customer Segments")
        st.pyplot(plt)
    else:
        st.warning("Please select exactly 2 numerical features.")
