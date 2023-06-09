#!/usr/bin/python
# import loader
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from os import walk
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordTokenizer

path = os.path.dirname(os.path.abspath(__file__)) + "\\"
data_path = path + 'data\\'
scores_path = path + 'scores\\'
samples_path = path + 'samples\\'

stopwords = stopwords.words('english')


def photos(execute, movies):
    if execute:
        allImagesArray = np.load(path + "Photos.npy", allow_pickle=True)
        arrayOfMovies = np.zeros((len(movies), 3))
        imageArray = np.array(allImagesArray[movies][:, 1])

        for n, imag in enumerate(tqdm(imageArray)):
            img = np.mean(imag, axis=(0, 1))
            arrayOfMovies[n] = img

        return arrayOfMovies


def text(execute, movies, maxFeatures):
    if execute:
        plot_array = np.load(path + "Plots_WithoutArray.npy", allow_pickle=True)
        moviesPlots = []

        for movie in movies:
            t = TreebankWordTokenizer()
            tokens = t.tokenize(plot_array[movie][1])
            content = [w for w in tokens if w.lower() not in stopwords]
            plot_array[movie][1] = ' '.join(content)
            moviesPlots.append(plot_array[movie])

        moviesPlots = np.array(moviesPlots, dtype=object)

        vectorizer = TfidfVectorizer(max_features=maxFeatures)
        response = vectorizer.fit_transform(moviesPlots[:, 1])
        return response.toarray()


def getMoviesWithSpecificGenres(genresList):
    ############
    # Get movies with specific genres but without duplicates in both genres
    # Works only if the genresList == 2
    ############

    if len(genresList) == 1:
        getMoviesWithSpecificGenre(genresList)
        exit()
    elif len(genresList) > 2:
        raise Exception('getMoviesWithSpecificGenres cannot have more than two genres')
    elif len(genresList) < 1:
        raise Exception('getMoviesWithSpecificGenres got null list')

    uniqueGenres = np.load(path + 'Y_UniqueGenres.npy', allow_pickle=True)
    indexOfGenres = []
    labels = []
    movies = []

    for genre in genresList:
        uniqueGenresFound = np.where(uniqueGenres == genre)
        if uniqueGenresFound[0].shape[0] != 0:
            indexOfGenres.append(uniqueGenresFound[0][0])

    # print(str(genresList) + ' -> ' + str(indexOfGenres))
    genresArray = np.load(path + 'Y_GenresArray.npy', allow_pickle=True)

    for i, genre in enumerate(genresArray):
        genresFound = []
        for index in indexOfGenres:
            if genre[index] == 1:
                genresFound.append(index)
        if len(genresFound) == len(indexOfGenres):
            continue  # skip movies with both genres
        for index in genresFound:
            labels.append(index)
            movies.append(i)

    print('Movies found: ' + str(len(movies)))
    unique, counts = np.unique(labels, return_counts=True)
    samples = list(zip(unique, counts))
    return indexOfGenres, movies, labels, samples


def getMoviesWithSpecificGenre(genre):
    ############
    # Works if the genresList == 1
    ############

    if len(genre) != 1:
        raise Exception("getMoviesWithSpecificGenre method works only for 1 genre")

    uniqueGenres = np.load(path + 'Y_UniqueGenres.npy', allow_pickle=True)
    indexOfGenre = np.where(uniqueGenres == genre)[0][0]

    genresArray = np.load(path + 'Y_GenresArray.npy', allow_pickle=True)
    labels = []
    movies = []

    for i, movie_genres in enumerate(genresArray):
        if movie_genres[indexOfGenre] == 1:
            labels.append(indexOfGenre)
            movies.append(i)

    print('Movies found: ' + str(len(movies)))
    unique, counts = np.unique(labels, return_counts=True)
    samples = list(zip(unique, counts))
    return indexOfGenre, movies, labels, samples


def getMoviesWithRandomGenres(genresArray, genresSeparated):
    labels = []
    movies = []

    for i, genre in enumerate(genresArray):
        genresFound = []
        for index in genresSeparated:
            if genre[index] == 1:
                genresFound.append(index)
        if len(genresFound) == len(genresSeparated):
            continue  # skip movies with both genres
        for index in genresFound:
            labels.append(index)
            movies.append(i)

    print('Movies found: ' + str(len(movies)))
    unique, counts = np.unique(labels, return_counts=True)
    samples = list(zip(unique, counts))
    return genresSeparated, movies, labels, samples


def prepareNewRandomGenres(N):
    ############
    # This has a drawback, sometimes can generate the two genres where there is
    # not common movies or there is a small group of them.
    # Has to be changed later
    ############
    firstRandomGenres = random.sample(range(27), N)
    secondRandomGenres = random.sample(range(27), N)
    genresArray = np.load(path + 'Y_GenresArray.npy', allow_pickle=True)

    for i in tqdm(range(N)):
        genresSeparated = [firstRandomGenres[i], secondRandomGenres[i]]
        genresIndexes, moviesGenresIndexesArray, y, samples = getMoviesWithRandomGenres(genresArray, genresSeparated)
        var1 = text(True, moviesGenresIndexesArray, 100)
        var2 = photos(True, moviesGenresIndexesArray)
        X = np.hstack([var1, var2])
        y = LabelEncoder().fit_transform(y)
        saveNpyFiles(X, y, samples, genresIndexes)


def prepareNpyFilesTextOrPhotos(genres_incl, text_incl, photos_incl):
    if not text_incl and not photos_incl:
        raise Exception('One parameter should be True [text_incl, photos_incl]')

    genreIndex, moviesGenresIndexesArray, y, samples = getMoviesWithSpecificGenres(genres_incl)
    var1 = text(text_incl, moviesGenresIndexesArray, 100)
    var2 = photos(photos_incl, moviesGenresIndexesArray)
    str_npy = ''

    if text_incl and photos_incl:
        X = np.hstack([var1, var2])
        str_npy = ''
    if text_incl:
        X = var1
        str_npy = 'text'
    elif photos_incl:
        X = var2
        str_npy = 'photos'

    y = LabelEncoder().fit_transform(y)
    saveNpyFiles(X, y, samples, genreIndex, extra_string=str_npy)


def saveNpyFiles(xValue, yValue, fileSamples, genresIndexList, extra_string=''):
    filename = '_'.join(str(i) for i in genresIndexList)
    with open(data_path + 'X_' + extra_string + '_' + filename + '.npy', 'wb') as f:
        np.save(f, xValue)
    with open(data_path + 'y_' + extra_string + '_' + filename + '.npy', 'wb') as f:
        np.save(f, yValue)
    with open(samples_path + 'sample_' + extra_string + '_' + filename + '.npy', 'wb') as f:
        np.save(f, fileSamples)


def manualGenres():
    ############
    # Not used anymore
    ############

    Genres = ['Sci-Fi', 'Crime']
    genresIndexes, moviesGenresIndexesArray, samples, y = getMoviesWithSpecificGenres(Genres)
    var1 = text(True, moviesGenresIndexesArray, 100)
    var2 = photos(True, moviesGenresIndexesArray)
    X = np.hstack([var1, var2])
    y = LabelEncoder().fit_transform(y)
    saveNpyFiles(X, y, samples, genresIndexes)


def checkSamples():
    samplesList = []
    samplesDirection = samples_path
    filenames = next(walk(samplesDirection), (None, None, []))[2]
    for i, file in enumerate(filenames):
        samplesList.append((file, np.load(samplesDirection + file, allow_pickle=True)))
        # print(samplesList[i])
    # print(len(samplesList))
    return samplesList


def createFilesFromRandomGenresCreated(text_incl, photos_incl):
    uniqueGenres = np.load(path + 'Y_UniqueGenres.npy', allow_pickle=True)
    print(uniqueGenres)
    genresNumberArray = []

    samplesList = checkSamples()
    for i, sample in tqdm(enumerate(samplesList)):
        matches = re.findall(r'\d+', samplesList[i][0])
        numbers = [int(match) for match in matches]
        prepareNpyFilesTextOrPhotos(genres_incl=uniqueGenres[numbers], text_incl=text_incl, photos_incl=photos_incl)


if __name__ == '__main__':
    print('prepare main')
    genres = ['Action', 'Sci-Fi']


    # getMoviesWithSpecificGenres(genres)
    # prepareNewRandomGenres(20)
    createFilesFromRandomGenresCreated(text_incl=False, photos_incl=True)
