import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import RandomOverSampler
from scipy import stats
from tqdm import tqdm
from tabulate import tabulate
from os import walk
import os

path = os.path.dirname(os.path.abspath(__file__)) + "\\"
data_path = path + 'data\\'
scores_path = path + 'scores\\'
samples_path = path + 'samples\\'
data_path_two = path + 'data2\\'
scores_path_two = path + 'scores2\\'

classifiers = [MLPClassifier(random_state=1, max_iter=300),
               KNeighborsClassifier(n_neighbors=3),
               DecisionTreeClassifier(),
               SVC(),
               GaussianNB()]

kf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1234)
ros = RandomOverSampler(random_state=24)


def classificationTwoGenres():
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

    scores = np.zeros((len(xList), len(classifiers), 10))

    for i in tqdm(range(len(xList))):
        X = np.load(data_path + xList[i], allow_pickle=True)
        y = np.load(data_path + yList[i], allow_pickle=True)

        try:
            for fold_index, (train_index, test_index) in enumerate(tqdm(kf.split(X, y))):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                X_res_train, y_res_train = ros.fit_resample(X_train, y_train)

                for cls_index, base_cls in enumerate(classifiers):
                    cls = clone(base_cls)
                    cls.fit(X_res_train, y_res_train)
                    y_pred = cls.predict(X_test)
                    score = balanced_accuracy_score(y_test, y_pred)
                    scores[i, cls_index, fold_index] = score
        except KeyError:
            print('Error! for data:' + str(xList[i]) + ' and ' + str(yList[i]))

    with open(scores_path + 'scores_' + str(len(xList)) + '_v3.npy', 'wb') as f:
        np.save(f, scores)

    scores_mean = np.mean(scores, axis=2)
    print(np.around(scores_mean, decimals=3))
    print(tabulate([scores_mean]))


def classificationTextOrPhotos(text_incl=False, photos_incl=False):
    if not text_incl and not photos_incl:
        raise Exception('One parameter should be True [text_incl, photos_incl]')

    xList = []
    yList = []
    filenames = next(walk(data_path_two), (None, None, []))[2]
    fileName = ''

    if text_incl and photos_incl:
        fileName = '_'
    if text_incl:
        fileName = '_text_'
    elif photos_incl:
        fileName = '_photos_'

    for file in filenames:
        if file.startswith("X" + fileName):
            xList.append(file)
        if file.startswith("y" + fileName):
            yList.append(file)

    yList.sort()

    if len(xList) != len(yList):
        print("ERROR! X LIST IS DIFFERENT THAN Y LIST")
        return

    scores = np.zeros((len(xList), len(classifiers), 10))

    for i in tqdm(range(len(xList))):
        print(xList[i])
        X = np.load(data_path_two + xList[i], allow_pickle=True)
        y = np.load(data_path_two + yList[i], allow_pickle=True)

        try:
            for fold_index, (train_index, test_index) in enumerate(tqdm(kf.split(X, y))):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                X_res_train, y_res_train = ros.fit_resample(X_train, y_train)

                for cls_index, base_cls in enumerate(classifiers):
                    cls = clone(base_cls)
                    cls.fit(X_res_train, y_res_train)
                    y_pred = cls.predict(X_test)
                    score = balanced_accuracy_score(y_test, y_pred)
                    scores[i, cls_index, fold_index] = score
        except KeyError:
            print('Error! for data:' + str(xList[i]) + ' and ' + str(yList[i]))

    with open(scores_path_two + 'scores' + fileName + 'v3.npy', 'wb') as f:
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


def got_mapped_matrix(array):
    mapping = {0: 'MLP', 1: 'KNN', 2: 'DTC', 3: 'SVC', 4: 'GNB'}
    listOfTrue = np.argwhere(array)

    new_list, temp = [], listOfTrue[-1]
    for item in range(0, temp[0]+1):
        new_list.append([[x[1], x[2]] for x in listOfTrue if x[0] == item])

    return [[(mapping[x[0]], mapping[x[1]]) for x in sublist] for sublist in new_list]


def statistics(scoresDone):
    alfa = 0.05
    scores_mean = np.mean(scoresDone, axis=2)
    print(np.around(scores_mean, decimals=3))
    mean_scores = np.mean(scores_mean, axis=0)
    print("Scores mean:", np.around(mean_scores, decimals=3))

    # mean ranks
    ranks = []
    for scores in scores_mean:
        ranks.append(stats.rankdata(scores).tolist())
    ranks = np.array(ranks)
    print("Ranks:", ranks)

    mean_ranks = np.mean(ranks, axis=0)
    print("Mean ranks:", mean_ranks)

    alfa = .05
    w_statistic = np.zeros((len(classifiers), len(classifiers)))
    p_rank_value = np.zeros((len(classifiers), len(classifiers)))

    for i in range(len(classifiers)):
        for j in range(len(classifiers)):
            w_statistic[i, j], p_rank_value[i, j] = stats.ranksums(ranks.T[i], ranks.T[j])

    significantlyBetterRank = np.logical_and(w_statistic > 0, p_rank_value <= alfa)
    indexed_matrix_ranks = got_mapped_matrix([significantlyBetterRank])

    # classifiers
    t_statistic = np.zeros((20, len(classifiers), len(classifiers)))
    p_value = np.zeros((20, len(classifiers), len(classifiers)))

    for i in range(scoresDone.shape[0]):
        # DATASETS
        for j in range(scoresDone.shape[1]):
            # CLASSIFIERS
            for k in range(scoresDone.shape[1]):
                t_statistic[i, j, k], p_value[i, j, k] = t_test_corrected(scoresDone[i, j], scoresDone[i, k])

    significantlyBetterStatArray = np.logical_and(t_statistic > 0, p_value <= alfa)
    indexed_matrix_clfs = got_mapped_matrix(significantlyBetterStatArray)

    return indexed_matrix_ranks, indexed_matrix_clfs


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


def generate_latex_code(scoresDone, table):
    latex_generated = []
    scores_mean = np.mean(scoresDone, axis=2)

    for n, table_element in enumerate(table):
        latex_acc = " & ".join(["{}".format(str(format(x, '.3f'))) for x in scores_mean[n]]) + ' \\\\'
        latex_generated.append(latex_acc)

        mapping_in = {'MLP': [], 'KNN': [], 'DTC': [], 'SVC': [], 'GNB': []}
        mapping_out = {'MLP': 1, 'KNN': 2, 'DTC': 3, 'SVC': 4, 'GNB': 5}

        for pair in table_element:
            mapping_in[pair[0]].append(mapping_out[pair[1]])

        latex_labels = '& & & '
        for key in mapping_in:
            if len(mapping_in[key]) == 0:
                values = "$^-$ & "
                latex_labels = latex_labels + values
            else:
                values = " ".join(["$^{}$".format(str(x)) for x in mapping_in[key]])
                latex_labels = latex_labels + values
                if str(key) != str(list(mapping_in.keys())[-1]):
                    latex_labels = latex_labels + ' & '
                else:
                    latex_labels = latex_labels + ' \\\\ '

        latex_generated.append(latex_labels)

    with open(scores_path_two + 'latex_wilcoxon.txt', 'w') as fp:
        fp.write("\n".join(str(item) for item in latex_generated))


if __name__ == '__main__':
    # classificationTwoGenres()
    # classificationTextOrPhotos(text_incl=False, photos_incl=True)

    # scoresDone = np.load(scores_path_two + 'scores_photos_v3.npy', allow_pickle=True)
    scoresDone = np.load(scores_path + 'scores_20_v3.npy', allow_pickle=True)
    ranks_table, clfs_table = statistics(scoresDone)
    print(ranks_table)
    # generate_latex_code(scoresDone, clfs_table)
