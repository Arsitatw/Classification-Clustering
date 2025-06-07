import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class GHIClustering:
    def __init__(self, data):
        self.data = data
        self.pca_data = None
        self.labels_ = None

    def apply_pca(self, n_components=2):
        pca = PCA(n_components=n_components)
        self.pca_data = pca.fit_transform(self.data)

    def kmeans_cluster(self, n_clusters=3):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.labels_ = kmeans.fit_predict(self.pca_data)
        
    def plot_clusters(self):
        if self.pca_data is None or self.labels_ is None:
            print("PCA data or cluster labels not found, please run apply_pca and kmeans_cluster first.")
            return
        
        plt.scatter(self.pca_data[:, 0], self.pca_data[:, 1], c=self.labels_, cmap='viridis', s=50)
        plt.title('KMeans Clustering on PCA-Reduced Data')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.colorbar(label='Cluster')
        plt.show()
