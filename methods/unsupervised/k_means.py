import pandas as pd
from sklearn.cluster import KMeans


data_kmeans = pd.read_excel('methods/datas/donnes_kmeans.xlsx')

def train_k_means_model():
    
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data_kmeans)
    centres = kmeans.cluster_centers_
    labels = kmeans.labels_

    return (centres, labels)
