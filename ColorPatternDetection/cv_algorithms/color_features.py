import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

example_image_path = '/mnt/captures/studies/pilots/sociopticon/Aug18/s--20210823--1323--0000000--pilot--patternCloth/undistorted/cam401539/image6000.png'
save_path = '/mnt/home/oshrihalimi/color_pattern_detection/'
save_dir = 'initial_tests'

from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def bench_k_means(kmeans, name, data, labels):
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(data, estimator[-1].labels_,
                                 metric="euclidean", sample_size=300,)
    ]

    # Show the results
    formatter_result = ("{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}"
                        "\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}")
    print(formatter_result.format(*results))

if __name__ == '__main__':
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, save_dir), exist_ok=True)

    image = cv2.imread(example_image_path)
    cv2.imwrite(os.path.join(save_path, save_dir, 'image.png'), image)

    # cut islands
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(save_path, save_dir, 'gray.png'), gray[..., None])

    blurred = cv2.GaussianBlur(gray, (5, 5), 3)
    cv2.imwrite(os.path.join(save_path, save_dir, 'blurred.png'), blurred[..., None])

    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
    cv2.imwrite(os.path.join(save_path, save_dir, 'laplacian.png'), 255 - 255 * (laplacian > 50))

    islands = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY_INV)[1]
    image = image * (islands[:, :, None] > 0)

    # calc color filters
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite(os.path.join(save_path, save_dir, 'H.png'), hsv[:, :, 0][..., None])
    cv2.imwrite(os.path.join(save_path, save_dir, 'S.png'), hsv[:, :, 1][..., None])
    cv2.imwrite(os.path.join(save_path, save_dir, 'V.png'), hsv[:, :, 2][..., None])

    # cyan
    corrected_color_cyan = hsv
    corrected_color_cyan[:, :, 0] = int(180/360*255)
    corrected_color_cyan = cv2.cvtColor(corrected_color_cyan, cv2.COLOR_HSV2BGR) * (islands[:, :, None] > 0)

    # to separate white - red image
    corrected_color_red = hsv
    corrected_color_red[:, :, 0] = 0
    corrected_color_red = cv2.cvtColor(corrected_color_red, cv2.COLOR_HSV2BGR)* (islands[:, :, None] > 0)

    # to separate white - yellow image
    corrected_color_yellow = hsv
    corrected_color_yellow[:, :, 0] = int(60/360*255)
    corrected_color_yellow = cv2.cvtColor(corrected_color_yellow, cv2.COLOR_HSV2BGR)* (islands[:, :, None] > 0)

    # to separate white - green image
    corrected_color_green = hsv
    corrected_color_green[:, :, 0] = int(120/360*255)
    corrected_color_green = cv2.cvtColor(corrected_color_green, cv2.COLOR_HSV2BGR)* (islands[:, :, None] > 0)

    # to separate white - blue image
    corrected_color_blue = hsv
    corrected_color_blue[:, :, 0] = int(240/360*255)
    corrected_color_blue = cv2.cvtColor(corrected_color_blue, cv2.COLOR_HSV2BGR)* (islands[:, :, None] > 0)

    # to separate white - magenta image
    corrected_color_magenta = hsv
    corrected_color_magenta[:, :, 0] = int(300/360*255)
    corrected_color_magenta = cv2.cvtColor(corrected_color_magenta, cv2.COLOR_HSV2BGR)* (islands[:, :, None] > 0)

    cv2.imwrite(os.path.join(save_path, save_dir, 'corrected_color_cyan.png'), corrected_color_cyan)
    cv2.imwrite(os.path.join(save_path, save_dir, 'corrected_color_red.png'), corrected_color_red)
    cv2.imwrite(os.path.join(save_path, save_dir, 'corrected_color_yellow.png'), corrected_color_yellow)
    cv2.imwrite(os.path.join(save_path, save_dir, 'corrected_color_green.png'), corrected_color_green)
    cv2.imwrite(os.path.join(save_path, save_dir, 'corrected_color_blue.png'), corrected_color_blue)
    cv2.imwrite(os.path.join(save_path, save_dir, 'corrected_color_magenta.png'), corrected_color_magenta)

    #################
    data = image.reshape((-1, 3))
    n_digits = 8
    reduced_data = data
    kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4)
    kmeans.fit(data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(data)
    cv2.imwrite(os.path.join(save_path, save_dir, 'labels.png'), Z.reshape(30, 30, 1) * 40)



    print("Hi")
