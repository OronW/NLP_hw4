import os

import gensim
from gensim.models import Word2Vec
import gc
from gensim.models import KeyedVectors
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold


path = r'C:\Users\oron.werner\PycharmProjects\NLP\hw4Inputs\allCountryFiles'
inputAllCountries = r'C:\Users\oron.werner\PycharmProjects\NLP\hw4Inputs\allCountryFiles\SumOfAll.txt'
hw4Inputs = r'C:\Users\oron.werner\PycharmProjects\NLP\hw4Inputs'
pathSelfTrained = r'C:\Users\oron.werner\PycharmProjects\NLP\hw4Inputs\self_trained_model.vec'
pathPreTrained = r'C:\Users\oron.werner\PycharmProjects\NLP\hw4Inputs\wiki.en.100k.vec'
allUsersOfCountry = r'C:\Users\oron.werner\PycharmProjects\NLP\hw4Inputs\allUsersOfCountryPhaseB'
chunkPath = r'C:\Users\oron.werner\PycharmProjects\NLP\hw4Inputs\allUsersOfCountryPhaseB\chunk'


def main():

    # combineSentences(allUsersOfCountry)
    totalCorpus = readAndLabel(chunkPath)
    model = KeyedVectors.load_word2vec_format(pathPreTrained, binary=False)


    createFeatureVectors(totalCorpus, 'LR', model, weightMod='random')
    createFeatureVectors(totalCorpus, 'LR', model)

    # if 'and' in model.vocab:
    #     print('HURRAY!')
    #
    # if 'etgf' in model.vocab:
    #     print('haa!')
    # print(model.word_vec('and')[:2])
    # print(model.get_vector('and')[:2])


@ignore_warnings(category=ConvergenceWarning)
def createFeatureVectors(totalCorpus, classifier, model, featureList=None, vectorType='normal', weightMod=None):

    totalDf = pd.DataFrame.from_dict(totalCorpus)     # create a data frame for the labeled sentences
    y = totalDf['class']     # create a column for of the labels
    X = totalDf['text']

    print(y.shape)
    print(X.shape)

    # print(X[0], ' ', y[0])
    # print(X[40000], ' ', y[40000])
    myVectorXtrain = []
    sumOfVec = np.zeros(shape=(300,))

    # print('Test random:')
    # ran = np.random.rand()
    # print('Random num: ', ran)
    # print(model.word_vec('and'))
    # result = model.word_vec('and') * ran
    # print('Result sum:')
    # print(result)

    total_accuracy_score = 0
    total_precision_score = 0
    total_recall_score = 0
    total_f1_score = 0

    if weightMod == 'random':
        print('Random weight')
        for sentence in X:
            for word in sentence.split():
                if word in model.vocab:
                    vecCalc = model.word_vec(word) * np.random.rand()
                    sumOfVec += vecCalc
            sumOfVec /= len(sentence.split())
            myVectorXtrain.append(sumOfVec.tolist())

    else:
        weight = 1

        for sentence in X:
            for word in sentence.split():
                if word in model.vocab:
                    vecCalc = model.word_vec(word) * weight
                    sumOfVec += vecCalc
            sumOfVec /= len(sentence.split())
            myVectorXtrain.append(sumOfVec.tolist())

    # gc.collect()

    myVectorXtrain = np.array(myVectorXtrain)
    print(myVectorXtrain.shape)

    lr = LogisticRegression(max_iter=100)
    cv_results = cross_validate(lr, myVectorXtrain, y, cv=10, scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'])
    print(cv_results.keys())
    print(cv_results['test_accuracy'])
    print(cv_results['test_precision_weighted'])
    print(cv_results['test_recall_weighted'])
    print(cv_results['test_f1_weighted'])


# -----------------------------------------
#     skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     print(skf.get_n_splits(X, y))
#
#     for train_index, test_index in skf.split(X, y):
#         print("\nTRAIN:", train_index, "TEST:", test_index)
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#
#
#         if vectorType == 'manual':
#             print('in manual')
#
#             myVectorXtrain = []
#             myVectorXtest = []
#             sentenceVector = []
#             sumOfVec = np.zeros(shape=(300,), dtype='float32')
#             sumOfVecTest = np.zeros(shape=(300,), dtype='float32')
#
#             TrainCounter = 0
#             TestCounter = 0
#
#             for sentence in X_train:
#                 # print('Train Counter: ', TrainCounter)
#                 # TrainCounter += 1
#                 # print(sentence)
#                 for word in sentence.split():
#                     # print(word)
#                     if word in model.vocab:
#                         vecCalc = model.word_vec(word) * weight
#                         # print(model.word_vec(word)[0])
#
#                         # print(len(sentence.split()))
#                         # print('vecCalc: ', vecCalc[0])
#                         sumOfVec += vecCalc
#                 sumOfVec /= len(sentence.split())
#                 # print(sumOfVec[0])
#                 myVectorXtrain.append(sumOfVec.tolist())
#                 # print(np.array(sumOfVec.tolist()).shape)
#                 # print(myVectorXtrain)
#
#                 # print(np.array(myVectorXtrain).shape)
#
#
#             for sentence in X_test:
#                 # print('Test Counter: ', TestCounter)
#                 # TestCounter += 1
#                 # print(sentence)
#                 for word in sentence.split():
#                     # print(word)
#                     if word in model.vocab:
#                         vecCalc = model.word_vec(word) * weight
#                         # print(model.word_vec(word)[0])
#
#                         # print(len(sentence.split()))
#                         # print('vecCalc: ', vecCalc[0])
#                         sumOfVecTest += vecCalc
#                 sumOfVecTest /= len(sentence.split())
#                 # print(sumOfVecTest[0])
#                 myVectorXtest.append(sumOfVecTest.tolist())
#
#             gc.collect()
#
#             X_train_dtm = np.array(myVectorXtrain, dtype='float32')  # create document - term matrix for the words
#             X_test_dtm = np.array(myVectorXtest, dtype='float32')
#
#             X_train_dtm = X_train_dtm.astype('float32')
#             X_test_dtm = X_test_dtm.astype('float32')
#
#             print(X_train_dtm.shape)
#             print(X_test_dtm.shape)
#
#
#         # -- this part for LR classifier --
#         if classifier == 'LR':
#             lr = LogisticRegression(max_iter=500)
#             lr.fit(X_train_dtm, y_train)
#             y_pred_class = lr.predict(X_test_dtm)
#
#             total_accuracy_score += metrics.accuracy_score(y_test, y_pred_class)
#             total_precision_score += metrics.precision_score(y_test, y_pred_class, average='micro')
#             total_recall_score += metrics.recall_score(y_test, y_pred_class, average='micro')
#             total_f1_score += metrics.f1_score(y_test, y_pred_class, average='micro')
#
#             print('\naccuracy_score: ', metrics.accuracy_score(y_test, y_pred_class))
#             print('precision_score: ', metrics.precision_score(y_test, y_pred_class, average='micro'))
#             print('recall_score: ', metrics.recall_score(y_test, y_pred_class, average='micro'))
#             print('f1_score: ', metrics.f1_score(y_test, y_pred_class, average='micro'))
#
#         else:
#             print('NO CLASSIFIER SELECTED FOR \'createFeatureVectors\' FUNCTION. ENDING RUN! ')
#             exit()
#
#
#         print('Test sentences by classes:')
#         print(y_test.value_counts())
#
#         # print('\naccuracy_score: ', metrics.accuracy_score(y_test, y_pred_class))
#         # print('\nprecision_score: ', metrics.precision_score(y_test, y_pred_class))
#         # print('\nrecall_score: ', metrics.recall_score(y_test, y_pred_class))
#         # print('\nf1_score: ', metrics.f1_score(y_test, y_pred_class))
#         # print('Confusion matrix:\n', metrics.confusion_matrix(y_test, y_pred_class))
#
#     print('\n**********************************')
#
#     # acc = sum/10
#     # total = int(sum*1000)/100
#     total_accuracy_score = int(total_accuracy_score*1000)/100
#     total_precision_score = int(total_precision_score*1000)/100
#     total_recall_score = int(total_recall_score*1000)/100
#     total_f1_score = int(total_f1_score*1000)/100
#
#     if classifier == 'LR':
#         print('Logistic Regression: ')
#         print('total_accuracy_score: ', total_accuracy_score)
#         print('total_precision_score: ', total_precision_score)
#         print('total_recall_score: ', total_recall_score)
#         print('total_f1_score: ', total_f1_score)
#         # summary = str('Logistic Regression: ' + str(total))
#         # summaryToFile.append(summary)



# TODO: make part of combinedSentences?
def readAndLabel(directory):

    label = 0
    totalCorpus = []
    fileCorpus = []

    # TODO: remove list limitations
    for currentFile in os.listdir(directory)[:]:
        if currentFile.endswith(".txt"):
            path1 = directory + '\\' + currentFile
            # print()
            # print('Reading the file: ')
            # print(path)

            f = open(path1, 'r', encoding='utf-8')
            sentences = f.read().splitlines()

            for sentence in sentences:
                totalCorpus.append({'text': sentence, 'class': label})

        label += 1

    return totalCorpus


def combineSentences(folderPath):

    largeSentence = ''

    for currentFile in os.listdir(folderPath):
        if currentFile.endswith(".txt"):
            path1 = folderPath + '\\' + currentFile
            # print()
            # print('Reading the file: ')
            # print(path)

            f = open(path1, 'r', encoding='utf-8')
            sentences = f.read().splitlines()

            newPath = folderPath + '\\chunk\\' + 'combined20' + currentFile

            f = open(newPath, 'w', encoding='utf-8')
            counter = 0

            for sentence in sentences:
                counter += 1
                largeSentence += sentence + ' '

                if counter == 20:
                    if sentence == sentences[len(sentences) - 1]:
                        f.write(largeSentence)
                    else:
                        f.write(largeSentence + '\n')
                        largeSentence = ''
                        counter = 0


def phaseA():

    model_pre_trained = KeyedVectors.load_word2vec_format(pathPreTrained, binary=False)
    model_self_trained = KeyedVectors.load_word2vec_format(pathSelfTrained, binary=False)

    print('pretty, beautiful ', model_pre_trained.similarity('pretty', 'beautiful'))
    print('ugly, beautiful ', model_pre_trained.similarity('ugly', 'beautiful'))
    print('pc, game', model_pre_trained.similarity('pc', 'game'))
    print('happy, birthday', model_pre_trained.similarity('happy', 'birthday'))
    print('king, queen', model_pre_trained.similarity('king', 'queen'))
    print()
    print('pretty, beautiful ', model_self_trained.similarity('pretty', 'beautiful'))
    print('ugly, beautiful ', model_self_trained.similarity('ugly', 'beautiful'))
    print('pc, game', model_self_trained.similarity('pc', 'game'))
    print('happy, birthday', model_self_trained.similarity('happy', 'birthday'))
    print('king, queen', model_self_trained.similarity('king', 'queen'))

    print('similar to game', model_pre_trained.most_similar('game'))
    print('similar to country', model_pre_trained.most_similar('country'))
    print('similar to learn', model_pre_trained.most_similar('learn'))

    print('similar to game', model_self_trained.most_similar('game'))
    print('similar to country', model_self_trained.most_similar('country'))
    print('similar to learn', model_self_trained.most_similar('learn'))

    print('calc nature + animal - water', model_pre_trained.most_similar(positive=['nature', 'animal'], negative='water'))
    print('calc movie + color - actor', model_pre_trained.most_similar(positive=['movie', 'color'], negative='actor'))

    print('calc nature + animal - water', model_self_trained.most_similar(positive=['movie', 'color'], negative='actor'))
    print('calc movie + color - actor', model_self_trained.most_similar(positive=['nature', 'animal'], negative='water'))


def oldMain():
    sentencesList = sentencesToListOfLists(inputAllCountries)
    if len(sentencesList) == 2214194:
        print('Sentences list created successfully')

    gc.collect
    self_trained_model = Word2Vec(sentencesList, size=300, min_count=10)
    self_trained_model.wv.save_word2vec_format(hw4Inputs + '\\' + 'self_trained_model')
    print('Finished')


def sentencesToListOfLists(inputPath):
    sentencesList = []
    lineList = []

    f = open(inputPath, 'r', encoding='utf-8')
    for line in f:
        splittedLine = line.rstrip().split()
        for word in splittedLine:
            lineList.append(word)
        sentencesList.append(lineList)
        lineList = []

    return sentencesList


# This function was used once to combine all country files into one
def combineFilesIntoOne(directory):
    totalCorpus = []

    for currentFile in os.listdir(directory):
        f = open(directory + '\\' + currentFile, 'r', encoding='utf-8')
        totalCorpus += f

    f = open(directory + "\\" + 'SumOfAll' + '.txt', 'w+', encoding='utf-8')
    for line in totalCorpus:
        f.write(line)



main()