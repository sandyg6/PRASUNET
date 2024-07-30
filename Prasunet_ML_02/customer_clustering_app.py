import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_data():
    data = pd.read_csv("Mall_Customers.csv")
    return data

def main():
    st.title("Customer Clustering App")
    st.write("This app uses KMeans clustering to group customers based on their annual income and spending score.")

    customer_data = load_data()
    # st.write("Here is a preview of the dataset:")
    # st.write(customer_data.head())

    st.sidebar.header("KMeans Clustering")
    n_clusters = st.sidebar.slider("Number of clusters", min_value=2, max_value=10, value=5)

    X = customer_data.iloc[:, [3, 4]].values

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
    Y = kmeans.fit_predict(X)

    plt.figure(figsize=(6, 6))
    colors = ['violet', 'cyan', 'orange', 'yellow', 'green', 'red', 'blue', 'brown', 'pink', 'gray']
    for i in range(n_clusters):
        plt.scatter(X[Y == i, 0], X[Y == i, 1], s=20, c=colors[i], label=f'Cluster {i+1}')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=45, c='black', label='Centroids')
    plt.title('Customer Groups')
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    plt.legend()
    st.pyplot(plt)

    st.sidebar.header("Input New Data")
    annual_income = st.sidebar.number_input("Annual Income", min_value=0, max_value=200, step=1)
    spending_score = st.sidebar.number_input("Spending Score", min_value=0, max_value=100, step=1)

    if st.sidebar.button("Predict Cluster"):
        new_data = np.array([[annual_income, spending_score]])
        new_cluster = kmeans.predict(new_data)
        st.sidebar.write(f"The new data point belongs to Cluster {new_cluster[0] + 1}")

        plt.figure(figsize=(8, 8))
        for i in range(n_clusters):
            plt.scatter(X[Y == i, 0], X[Y == i, 1], s=50, c=colors[i], label=f'Cluster {i+1}')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label='Centroids')
        plt.scatter(new_data[0, 0], new_data[0, 1], s=100, c='red', label='New Data Point', edgecolor='black')
        plt.title('Customer Groups')
        plt.xlabel('Annual Income')
        plt.ylabel('Spending Score')
        plt.legend()
        st.pyplot(plt)

if __name__ == '__main__':
    main()
