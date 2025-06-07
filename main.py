import os
from src.ghi_preprocessing import GHIDataPreprocessor
from src.clustering import GHIClustering
from src.classification import GHIClassifier
from src.visualisasi import plot_clusters, save_confusion_matrix

def main():
    # 1. Load dan preprocess data
    files = {
        'undernourishment': 'Dataset/Proportion of undernourished in the population.csv',
        'stunting': 'Dataset/Prevalence of stunting in children under five years.csv',
        'wasting': 'Dataset/Prevalence of wasting in children under five years.csv',
        'mortality': 'Dataset/Under-five mortality rate.csv',
        'ghi': 'Dataset/GHI2022 scores.csv'
    }

    prep = GHIDataPreprocessor(files)
    prep.load_and_merge()
    prep.scale_features()

    # 2. Clustering dengan PCA + KMeans
    clustering = GHIClustering(prep.scaled_data)
    clustering.apply_pca(n_components=2)
    clustering.kmeans_cluster(n_clusters=3)

    # Tambahkan hasil cluster ke DataFrame asli
    prep.df['Cluster'] = clustering.labels_

    # Visualisasi cluster hasil PCA dan KMeans
    plot_clusters(clustering.pca_data, clustering.labels_, show_plot=True)



    # 3. Classification Random Forest, klasifikasi cluster
    clf = GHIClassifier(prep.df, prep.features, 'Cluster')
    report, matrix = clf.train_random_forest()

    # Simpan hasil classification report
    os.makedirs("results", exist_ok=True)
    with open("results/classification_report.txt", "w") as f:
        f.write(report)

    # Simpan confusion matrix (misal dalam bentuk gambar)
    save_confusion_matrix(matrix, "results/confusion_matrix.png")

if __name__ == "__main__":
    main()
