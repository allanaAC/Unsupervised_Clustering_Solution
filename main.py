from src.data.load_dataset import load_and_preprocess_data
from src.visualization.visualize import plot_pairplot, visualize_clusters, plot_elbow, plot_silhouette
from src.model.train_model import kmeans_clustering, elbow_method, silhouette_analysis
from src.model.predict_model import predict_model
import warnings
import matplotlib.pyplot as plt
import pandas as pd
#warnings.filterwarnings("ignore")
#plt.style.use('ggplot')

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    #plt.style.use('ggplot')
    
    # Load the data
    df = load_and_preprocess_data('src/data/mall_customers.csv')
    print(df.head())

    # Plot pairplot
    plot_pairplot(df, ['Age', 'Annual_Income', 'Spending_Score'])

    # Perform K-means clustering
    features = ['Annual_Income', 'Spending_Score']
    model, df = kmeans_clustering(df, features, n_clusters=5)

    # Visualize clusters
    visualize_clusters(df, 'Annual_Income', 'Spending_Score', 'Cluster')

    # Elbow method
    k_range = range(3, 9)
    wss = elbow_method(df, features, k_range)
    plot_elbow(wss)

    # Silhouette analysis
    silhouette_scores = silhouette_analysis(df, features, k_range)
    wss['Silhouette_Score'] = silhouette_scores
    plot_silhouette(wss)

    # K-means with all features
    all_features = ['Age', 'Annual_Income', 'Spending_Score']
    silhouette_scores = silhouette_analysis(df, all_features, k_range)
    variables3 = pd.DataFrame({'cluster': k_range, 'Silhouette_Score': silhouette_scores})
    plot_silhouette(variables3)