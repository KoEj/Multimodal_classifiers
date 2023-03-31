#!/usr/bin/python
import threading
import numpy as np
import glob, os
import json
import csv
import cv2
import time
import matplotlib.pyplot as plt
databaseDirection = "C:/Users/PLUSR6000280/Documents/Studia/Uczenie maszynowe/mmimdb/mmimdb/dataset/"
# C:\Users\PLUSR6000280\Documents\Studia\Uczenie maszynowe\mmimdb\mmimdb\dataset
direction = "C:/Users/PLUSR6000280/PycharmProjects/Uczenie_Maszynowe/"

print('loader executed')
library = {}
genresArray = []
# yArray = []
plotArray = []
ExecuteGenres = False
ExecutePlots = False
ExecutePhotos = False
plotsJsonIdArray = []
photosArray = [] #ximg


def getGenresData(json_name):
    file = open(json_name)
    data = json.load(file)

    try:
        genresArray.append([json_name.split(".")[0], data['genres']])
    except KeyError:
        print(json_name + ' has KeyError! Check it')


def getPlotData(json_name):
    file = open(json_name)
    data = json.load(file)

    try:
        plotArray.append([json_name.split(".")[0], ' '.join(data['plot'])])
    except KeyError:
        print(json_name + ' has KeyError! Check it')


def getPhotoData(jpeg_name):
    img = cv2.imread(jpeg_name)
    img = cv2.resize(img, dsize)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    photosArray.append([jpeg_name.split(".")[0], img])

threads = list()
fileList = ["0000008.json", "0000005.json"]
os.chdir(databaseDirection)

#Genres Done
if ExecuteGenres:
    for item in glob.glob('*.json'):
    # for item in fileList:
        x = threading.Thread(target=getGenresData, args=(str(item),))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        thread.join()

    print(genresArray)
    genresArray.sort()

    allGenres = []
    for genre in genresArray:
        allGenres.append(genre[1])

    uniqueGenres = np.sort(np.unique(np.concatenate(allGenres)))
    # print(uniqueGenres)
    with open(direction + 'Y_UniqueGenres.npy', 'wb') as f:
        np.save(f, uniqueGenres)
    genresArrayYarray = np.zeros((len(allGenres), len(uniqueGenres)), dtype=int)
    # print(genresArrayYarray)
    # print(allGenres)

    for gen in enumerate(allGenres):
        for genre in gen[1]:
            genresArrayYarray[gen[0], np.where(uniqueGenres == genre)] = 1

    print(genresArrayYarray)
    with open(direction + 'Y_GenresArray.npy', 'wb') as f:
        np.save(f, genresArrayYarray)


if ExecutePlots:
    for item in glob.glob('*.json'):
    # for item in fileList:
        x = threading.Thread(target=getPlotData, args=(str(item),))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        thread.join()

    plotArray = np.array(plotArray, dtype=object)
    plotArray = plotArray[plotArray[:, 0].argsort()]

    with open(direction + 'Plots_WithoutArray.npy', 'wb') as f:
        np.save(f, plotArray)


if ExecutePhotos:
    fileList = ["0000012.jpeg", "0000008.jpeg"]
    dsize = (100, 100)
    for item in glob.glob('*.jpeg'):
    # for item in fileList:
        x = threading.Thread(target=getPhotoData, args=(str(item),))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        thread.join()

    photosArray = np.array(photosArray, dtype=object)
    photosArray = photosArray[photosArray[:, 0].argsort()]

    with open(direction + 'Photos.npy', 'wb') as f:
        np.save(f, photosArray)

