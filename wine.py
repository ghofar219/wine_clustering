import streamlit as st
import numpy as np 
import pandas as pd 
pd.set_option('display.max_column', None)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA

df = pd.read_csv('wine-clustering.csv')


X = df

st.header("isi dataset")
st.write(X)

# menampilkan panah elbow
k_values = list()
sse = list()

scaler = StandardScaler()

wine_scaled = scaler.fit_transform(df)

for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=0, n_init=10).fit(wine_scaled)
    k_values.append(i)
    sse.append(kmeans.inertia_)

ig, ax = plt.subplots(figsize=(8, 5))

ax.plot(k_values, sse)
ax.scatter(k_values, sse, c='r')

ax.set_title('Elbow Method', fontsize=15)
ax.set_xlabel('K-values', fontsize=13)
ax.set_ylabel('SSE', fontsize=13)

ax.spines[['top', 'right']].set_visible(False)

st.set_option('deprecation.showPyplotGlobalUse', False)
elbo_plot = st.pyplot()

st.sidebar.subheader("Nilai jumlah K")
clust = st.sidebar.slider("Pilih jumlah cluster :", 2,10,3,1)

def k_means(n_clust):
    kmean = KMeans(n_clusters=n_clust).fit(X)
    X['Labels'] = kmean.labels_
    
    pca = PCA(n_components=n_clust)

    X_pca = pca.fit_transform(wine_scaled)

    kmeans = KMeans(n_clusters=n_clust)

    clusters = kmeans.fit_predict(X_pca)

    for cluster_num in range(max(clusters) + 1):
        cluster_points = X_pca[clusters == cluster_num]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_num+1}')
    
    pca_df = pd.DataFrame({'PCA 1': X_pca[:, 0], 'PCA 2': X_pca[:, 1], 'Cluster': clusters})

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Clustering Result')
    plt.legend()
    plt.show()
            
    st.header('Cluster Plot')
    st.pyplot()
    st.write(pca_df)

k_means(clust)


