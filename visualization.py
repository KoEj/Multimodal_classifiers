from scipy.signal import medfilt2d
from tqdm import tqdm

from prepare import getMoviesWithSpecificGenres, getMoviesWithSpecificGenre
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import cv2
import math

path = os.path.dirname(os.path.abspath(__file__)) + "\\"
visualization_path = path + "visualization\\"

def getVisualization(N, genresList):
    uniqueGenres = np.load(path + 'Y_UniqueGenres.npy', allow_pickle=True)

    if len(genresList) == 1:
        genresSeparated, movies, labels, samples = getMoviesWithSpecificGenre(genresList)
    else:
        genresSeparated, movies, labels, samples = getMoviesWithSpecificGenres(genresList)

    plot_array = np.load(path + "Plots_WithoutArray.npy", allow_pickle=True)
    allImagesArray = np.load(path + "Photos250x300.npy", allow_pickle=True)
    wc = WordCloud(background_color="white", width=250, height=300)

    pad = 10
    width = N * (250 + pad) + pad
    height = 2 * (300 + pad) + pad

    f, ax = plt.subplots(2, N, figsize=(width / 100, height / 100))
    plt.figure(1)
    for i in range(N):
        ax[0, i].imshow(allImagesArray[movies[i]][1])
        ax[0, i].axis("off")
        ax[0, i].set_title(uniqueGenres[labels[i]])
        wordcloud = wc.generate(plot_array[movies[i]][1])
        ax[1, i].imshow(wordcloud, interpolation='bilinear')
        ax[1, i].axis("off")
    plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.5)
    plt.savefig(visualization_path + 'wordcloud_' + str('_'.join(genresList).lower()))


def find_integers(C):
    int1 = math.ceil(math.sqrt(C) // 2)
    int2 = math.ceil(C / int1)
    return int1, int2


def getColorMap(genresList):
    allImagesArray = np.load(path + "Photos.npy", allow_pickle=True)

    if len(genresList) == 1:
        genresSeparated, movies, labels, samples = getMoviesWithSpecificGenre(genresList)
    else:
        genresSeparated, movies, labels, samples = getMoviesWithSpecificGenres(genresList)

    moviesNumer = len(movies)
    first, second = find_integers(moviesNumer)
    arrayOfMovies = np.zeros((first*second, 3))
    imageArray = np.array(allImagesArray[movies][:, 1])

    for n, imag in enumerate(tqdm(imageArray)):
        img = np.mean(imag, axis=(0, 1))
        arrayOfMovies[n] = img

    images = arrayOfMovies.reshape(first, second, 3)

    for i in range(3):
        images[:, :, i] = medfilt2d(images[:, :, i], kernel_size=5)

    plt.figure(2, figsize=(11, 3))
    plt.imshow(images.astype(int))
    plt.tight_layout(pad=2)
    plt.suptitle(' '.join(genresList))
    plt.axis("off")
    plt.savefig(visualization_path + 'colormap_' + str('_'.join(genresList).lower()))


if __name__ == '__main__':
    print('visualization')
    genres = ['Action', 'Sci-Fi']

    getVisualization(10, genres)
    # getColorMap(genres)
