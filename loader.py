#!/usr/bin/python
import threading
import numpy as np
import glob, os
import json
import cv2

path = os.path.dirname(os.path.abspath(__file__)) + "\\"
data_path = path + 'data\\'
scores_path = path + 'scores\\'
samples_path = path + 'samples\\'

# path to the mmimdb dataset (can be downloaded from https://archive.org/download/mmimdb/mmimdb.tar.gz)
database_path = "C:\\Users\\PLUSR6000280\\Documents\\Studia\\Uczenie maszynowe\\mmimdb\\mmimdb\\dataset"
#

print('loader started execute')
library = {}
threads = list()
genresArray = plotArray = photosArray = []
os.chdir(database_path)

ExecuteGenres = False  # change value to got .npy files created
ExecutePlots = False
ExecutePhotos = True


def getGenresData(json_name):
    file = open(json_name)
    data = json.load(file)

    try:
        genresArray.append([json_name.split(".")[0], data['genres']])
    except KeyError:
        print(json_name + ' has KeyError! Check it')


if ExecuteGenres:
    for item in glob.glob('*.json'):
        x = threading.Thread(target=getGenresData, args=(str(item),))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        thread.join()

    genresArray.sort()
    allGenres = []
    for genre in genresArray:
        allGenres.append(genre[1])

    uniqueGenres = np.sort(np.unique(np.concatenate(allGenres)))
    with open(path + 'Y_UniqueGenres.npy', 'wb') as f:
        np.save(f, uniqueGenres)
    genresArrayYarray = np.zeros((len(allGenres), len(uniqueGenres)), dtype=int)

    for gen in enumerate(allGenres):
        for genre in gen[1]:
            genresArrayYarray[gen[0], np.where(uniqueGenres == genre)] = 1

    print(genresArrayYarray)
    with open(path + 'Y_GenresArray.npy', 'wb') as f:
        np.save(f, genresArrayYarray)


def getPlotData(json_name):
    file = open(json_name)
    data = json.load(file)

    try:
        plotArray.append([json_name.split(".")[0], ' '.join(data['plot'])])
    except KeyError:
        print(json_name + ' has KeyError! Check it')


if ExecutePlots:
    for item in glob.glob('*.json'):
        x = threading.Thread(target=getPlotData, args=(str(item),))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        thread.join()

    plotArray = np.array(plotArray, dtype=object)
    plotArray = plotArray[plotArray[:, 0].argsort()]

    with open(path + 'Plots_WithoutArray.npy', 'wb') as f:
        np.save(f, plotArray)


def getPhotoData(jpeg_name):
    img = cv2.imread(jpeg_name)
    img = cv2.resize(img, dsize)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    photosArray.append([jpeg_name.split(".")[0], img])


if ExecutePhotos:
    dsize = (250, 300)
    for item in glob.glob('*.jpeg'):
        x = threading.Thread(target=getPhotoData, args=(str(item),))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        thread.join()

    photosArray = np.array(photosArray, dtype=object)
    photosArray = photosArray[photosArray[:, 0].argsort()]

    with open(path + 'Photos250x300.npy', 'wb') as f:
        np.save(f, photosArray)

