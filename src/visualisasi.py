def plot_clusters(pca_data, labels, save_path=None, show_plot=False):
    import matplotlib.pyplot as plt
    import os
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='Set1', s=50)
    plt.title("Cluster Visualization after PCA")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True)
    plt.colorbar(scatter, label='Cluster Label')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    else:
        plt.close()

def save_confusion_matrix(matrix, save_path=None, show_plot=False):
    import matplotlib.pyplot as plt
    import os
    from sklearn.metrics import ConfusionMatrixDisplay

    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    else:
        plt.close()
