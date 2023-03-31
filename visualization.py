from prepare import getMoviesWithSpecificGenres, text, photos
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import cv2

path = os.path.dirname(os.path.abspath(__file__)) + "\\"

def getVisualization(N):
    uniqueGenres = np.load(path + 'Y_UniqueGenres.npy', allow_pickle=True)
    genresSeparated, movies, labels, samples = getMoviesWithSpecificGenres(['Animation', 'Music'])
    plot_array = np.load(path + "Plots_WithoutArray.npy", allow_pickle=True)
    allImagesArray = np.load(path + "Photos250x300.npy", allow_pickle=True)
    wc = WordCloud(background_color="white", width=250, height=300)

    pad = 10
    width = N * (250 + pad) + pad
    height = 2 * (300 + pad) + pad

    f, ax = plt.subplots(2, N, figsize=(width / 100, height / 100))
    for i in range(N):
        ax[0, i].imshow(allImagesArray[movies[i]][1])
        ax[0, i].axis("off")
        ax[0, i].set_title(uniqueGenres[labels[i]])
        wordcloud = wc.generate(plot_array[movies[i]][1])
        ax[1, i].imshow(wordcloud, interpolation='bilinear')
        ax[1, i].axis("off")
    plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.5)
    plt.savefig('wordcloud_with_text')


# def getColorMap():
#     allImagesArray = np.load(path + "Photos.npy", allow_pickle=True)
#     # for print purposes
#     # arrayOfMovies = np.zeros((5900, 3))
#     arrayOfMovies = np.zeros((len(movies), 3))
#     imageArray = np.array(allImagesArray[movies][:, 1])
#
#     for n, imag in enumerate(tqdm(imageArray)):
#         img = np.mean(imag, axis=(0, 1))
#         print(img)
#         arrayOfMovies[n] = img

    # imagess = arrayOfMovies.reshape((59, 100, 3))
    # for i in range(3):
    #     imagess[:, :, i] = medfilt2d(imagess[:, :, i], kernel_size=5)
    # plt.imshow(imagess.astype(int))
    # plt.show()

# for print purposes
# arrayOfMovies = np.zeros((5900, 3))

if __name__ == '__main__':
    print('visualization')
    getVisualization(10)