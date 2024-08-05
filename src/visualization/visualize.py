import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix

def plot_pairplot(df, features):
    try:
        sns.pairplot(df[features])
        plt.show() 
    except Exception as e:
        logging.error(" Error in plot_pairplot data: {}". format(e))
        
def visualize_clusters(df, x, y, hue):
    try:
        sns.scatterplot(x=x, y=y, data=df, hue=hue, palette='colorblind')
        plt.show()
    except Exception as e:
        logging.error(" Error in visualize_clusters data: {}". format(e))
        

def plot_elbow(wss):
    try:
        wss.plot(x='cluster', y='WSS_Score')
        plt.xlabel('No. of clusters')
        plt.ylabel('WSS Score')
        plt.title('Elbow Plot')
        plt.show()
    except Exception as e:
        logging.error(" Error in plot_elbow data: {}". format(e))
        

def plot_silhouette(wss):
    try:
        wss.plot(x='cluster', y='Silhouette_Score')
        plt.xlabel('No. of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Plot')
        plt.show()
    except Exception as e:
        logging.error(" Error in plot_silhouette data: {}". format(e))