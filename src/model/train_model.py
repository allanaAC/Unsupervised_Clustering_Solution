from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging
import pandas as pd


# Function to train the model
def kmeans_clustering(df, features, n_clusters):
    try:
        model = KMeans(n_clusters=n_clusters).fit(df[features])
        df['Cluster'] = model.labels_
        return model, df
    except Exception as e:
        logging.error(" Error in kmeans_clustering data: {}". format(e))
        
def elbow_method(df, features, k_range):
    try:
        wcss = []
        for k in k_range:
            model = KMeans(n_clusters=k).fit(df[features])
            wcss.append(model.inertia_)
        return pd.DataFrame({'cluster': k_range, 'WSS_Score': wcss})
    except Exception as e:
        logging.error(" Error in elbow_method data: {}". format(e))
        
def silhouette_analysis(df, features, k_range):
    try:
        silhouette_scores = []
        for k in k_range:
            model = KMeans(n_clusters=k).fit(df[features])
            score = silhouette_score(df[features], model.labels_)
            silhouette_scores.append(score)
        return silhouette_scores
    except Exception as e:
        logging.error(" Error in silhouette_analysis data: {}". format(e))
        