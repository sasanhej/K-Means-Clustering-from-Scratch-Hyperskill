import numpy as np
from sklearn.datasets import load_wine
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from sklearn.metrics import silhouette_score


# scroll down to the bottom to implement your solution


def plot_comparison(data: np.ndarray, predicted_clusters: np.ndarray, true_clusters: np.ndarray = None,
                    centers: np.ndarray = None, show: bool = True):

    # Use this function to visualize the results on Stage 6.

    if true_clusters is not None:
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Predicted clusters')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()

        plt.subplot(1, 2, 2)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=true_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Ground truth')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()
    else:
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Predicted clusters')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()

    plt.savefig('Visualization.png', bbox_inches='tight')
    if show:
        plt.show()

def find_nearest_center(centers: np.ndarray, features: np.ndarray):
    return [np.argmin([euclidean(feature, center) for center in centers]) for feature in features]


def calculate_new_centers(clusters):
    return [np.mean(c, axis=0) for c in clusters]
class CustomKMeans:
    def __init__(self, k):
        self.k = k
        self.centers = None
    def fit(self, X, eps=1e-6):
        norm = 1
        start_centers = np.array(X[0:self.k])
        while norm > eps:
            labels = find_nearest_center(start_centers, X_full)
            clusters = [[] for y in range(self.k)]
            for idx, x in enumerate(X_full):
                clusters[labels[idx]].append(x)
            new_centers = np.array(calculate_new_centers(clusters))
            norm = np.linalg.norm(start_centers - new_centers)
            start_centers = new_centers
        self.centers = start_centers
    def predict(self, X):
        return find_nearest_center(self.centers, X)

    def SQRS(self,X):
        titles=self.predict(X)
        centers=self.centers
        jam=sum([euclidean(X[i], centers[titles[i]])**2 for i in range(len(X))])
        return (jam)

    def slt(self,X):
        titles=self.predict(X)
        sltt=silhouette_score(X,titles)
        return sltt

    def plot(self,X):
        plot_comparison(X_full, np.array(self.predict(X_full)), None, self.centers,True)


if __name__ == '__main__':

    # Load data
    data = load_wine(as_frame=True, return_X_y=True)
    X_full, y_full = data

    # Permutate it to make things more interesting
    rnd = np.random.RandomState(42)
    permutations = rnd.permutation(len(X_full))
    X_full = X_full.iloc[permutations]
    y_full = y_full.iloc[permutations]

    # From dataframe to ndarray
    X_full = X_full.values
    y_full = y_full.values

    # Scale data
    scaler = StandardScaler()
    X_full = scaler.fit_transform(X_full)
    # write your code here
    rt=[]
    sl=[]
    k = 3
    means = CustomKMeans(k)
    means.fit(X_full)
    #rt.append(means.SQRS(X_full))
    sl.append(means.slt(X_full))
    #print(rt)
    #print(sl)
    print(means.predict(X_full[:20]))
    exit(0)