import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from scipy.stats import ttest_ind, rankdata
from scipy import stats
from tqdm import tqdm
from tabulate import tabulate
from os import walk
import os

path = os.path.dirname(os.path.abspath(__file__)) + "\\"
data_path = path + 'data\\'
scores_path = path + 'scores\\'
samples_path = path + 'samples\\'

classifiers = [MLPClassifier(random_state=1, max_iter=300),
               KNeighborsClassifier(n_neighbors=3),
               DecisionTreeClassifier(),
               SVC()]

kf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1234)

def classification():
    xList = []
    yList = []
    filenames = next(walk(data_path), (None, None, []))[2]
    for file in filenames:
        if file.startswith("X"):
            xList.append(file)
        if file.startswith("y"):
            yList.append(file)

    yList.sort()

    if len(xList) != len(yList):
        print("ERROR! X LIST IS DIFFERENT THAN Y LIST")
        return

    scores = np.zeros((len(xList), 4, 10))

    for i in tqdm(range(len(xList))):
        print(xList[i])
        X = np.load(data_path + xList[i], allow_pickle=True)
        y = np.load(data_path + yList[i], allow_pickle=True)

        try:
            for fold_index, (train_index, test_index) in enumerate(tqdm(kf.split(X, y))):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                for cls_index, base_cls in enumerate(classifiers):
                    cls = clone(base_cls)
                    cls.fit(X_train, y_train)
                    y_pred = cls.predict(X_test)
                    score = balanced_accuracy_score(y_test, y_pred)
                    scores[i, cls_index, fold_index] = score
        except KeyError:
            print('Error! for data:' + str(xList[i]) + ' and ' + str(yList[i]))

    with open(scores_path + 'scores_' + str(len(xList)) + '_v2.npy', 'wb') as f:
        np.save(f, scores)

    scores_mean = np.mean(scores, axis=2)
    print(np.around(scores_mean, decimals=3))
    print(tabulate([scores_mean]))



def checkSamples():
    samplesList = []
    filenames = next(walk(samples_path), (None, None, []))[2]
    for file in filenames:
        samplesList.append(file)

    samplesList.sort()
    uniqueGenres = np.load(path + 'Y_UniqueGenres.npy', allow_pickle=True)

    genresDifferenceArray = []
    for item in samplesList:
        itemSplitted = item.split('_')
        sample = np.load(samples_path + item, allow_pickle=True)
        sampleConcat = sample[:, 1]

        if sampleConcat[0] > sampleConcat[1]:
            ir = sampleConcat[0] / sampleConcat[1]
        else:
            ir = sampleConcat[1] / sampleConcat[0]

        genresDifferenceArray.append((uniqueGenres[int(itemSplitted[1])],
                                      uniqueGenres[int(itemSplitted[2].split('.')[0])],
                                      sampleConcat[0],
                                      sampleConcat[1],
                                      ir))

    print(genresDifferenceArray)
    return genresDifferenceArray


def statistics(N):
    scoresDone = np.load(scores_path + 'scores_20_v2.npy', allow_pickle=True)

    # Ranks
    scores_mean = np.mean(scoresDone, axis=2)
    print(np.around(scores_mean, decimals=3))
    ranks = []
    for ms in scores_mean:
        ranks.append(rankdata(ms).tolist())
    ranks = np.array(ranks)
    mean_ranks = np.mean(ranks, axis=0)
    print("\nMean ranks:\n", mean_ranks)

    alfa = 0.05
    t_statistic = p_value = np.zeros((20, 4, 4))

    for i in range(scoresDone.shape[0]):
        # DATASETS
        for j in range(scoresDone.shape[1]):
            # CLASSIFIERS
            for k in range(scoresDone.shape[1]):
                t_statistic[i, j, k], p_value[i, j, k] = t_test_corrected(scoresDone[i, j], scoresDone[i, k])

    significantlyBetterStatArray = np.logical_and(t_statistic > 0, p_value <= alfa)
    # print(significantlyBetterStatArray.astype(int))
    listOfTrue = np.argwhere(significantlyBetterStatArray)

    new_list, temp = [], listOfTrue[-1]
    for item in range(0, temp[0]+1):
        new_list.append([[x[1], x[2]] for x in listOfTrue if x[0] == item])

    mapping = {0: 'MLPC', 1: 'KNN', 2: 'DTC', 3: 'SVC'}

    indexed_matrix = [[(mapping[x[0]], mapping[x[1]]) for x in sublist] for sublist in new_list]

    for i in indexed_matrix:
        print(i)


def cv52cft(a, b, J=5, k=2):
    """
    Combined 5x2CV F test.
    input, two 2d arrays. Repetitions x folds
    As default for 5x2CV
    """
    if J*k != a.shape[0]:
        raise Exception('%i scores received, but J=%i, k=%i (J*k=%i)' % (
            a.shape[0], J, k, J * k
        ))

    d = a.reshape(k, J) - b.reshape(k, J)
    print(d)
    f_stat = np.sum(np.power(d, 2)) / (2 * np.sum(np.var(d, axis=0, ddof=0)))
    print(f_stat)
    p = 1 - stats.f.cdf(f_stat, J * k, J)
    print(p)
    return f_stat, p


def t_test_corrected(a, b, J=5, k=2):
    """
    Corrected t-test for repeated cross-validation.
    input, two 2d arrays. Repetitions x folds
    As default for 5x5CV
    """
    if J*k != a.shape[0]:
        raise Exception('%i scores received, but J=%i, k=%i (J*k=%i)' % (
            a.shape[0], J, k, J*k
        ))

    d = a - b
    bar_d = np.mean(d)
    bar_sigma_2 = np.var(d.reshape(-1), ddof=1)
    bar_sigma_2_mod = (1 / (J * k) + 1 / (k - 1)) * bar_sigma_2
    t_stat = bar_d / np.sqrt(bar_sigma_2_mod)
    pval = stats.t.sf(np.abs(t_stat), (k * J) - 1) * 2
    return t_stat, pval


if __name__ == '__main__':
    # scoresDone = np.load(directionScores + 'scores_20_v2', allow_pickle=True)
    # print(tabulate([scoresDone]))

    # classification()
    statistics(20)
    # checkSamples()
