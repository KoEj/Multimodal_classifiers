from prepare import getMoviesWithSpecificGenres, text, photos
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

direction = "C:/Users/PLUSR6000280/PycharmProjects/Uczenie_Maszynowe/"

def getVisualization(N):
    uniqueGenres = np.load(direction + 'Y_UniqueGenres.npy', allow_pickle=True)
    genresSeparated, movies, labels, samples = getMoviesWithSpecificGenres(['Animation', 'Music'])
    plot_array = np.load(direction + "Plots_WithoutArray.npy", allow_pickle=True)
    allImagesArray = np.load(direction + "Photos.npy", allow_pickle=True)

    f, ax = plt.subplots(2, N)
    for i in range(5):
        ax[0, i].imshow(allImagesArray[movies[i]][1])
        ax[0, i].axis("off")
        ax[0, i].set_title(uniqueGenres[labels[i]])
        wordcloud = WordCloud(background_color="white").generate(plot_array[movies[i]][1])
        ax[1, i].imshow(wordcloud, interpolation='bilinear')
        ax[1, i].axis("off")
    plt.tight_layout()
    plt.savefig('wordcloud_with_text')


getVisualization(5)