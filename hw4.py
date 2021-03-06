import os
import random
import sys
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
import datetime


summaryToFile = []


def main(chunkPath=sys.argv[1], pathPreTrained=sys.argv[2], pathSelfTrained=sys.argv[3], summaryOutputPath=sys.argv[4]):

    print(datetime.datetime.now())

    # combineSentences(topUsersOfCountry)

    totalCorpus = readAndLabel(chunkPath)
    shuffledTotalCorpus = totalCorpus.copy()
    random.shuffle(shuffledTotalCorpus)


    modelPreTrained = KeyedVectors.load_word2vec_format(pathPreTrained, binary=False)

    modelSelfTrained = KeyedVectors.load_word2vec_format(pathSelfTrained, binary=False)

    print('Arithmetic weights:')
    summaryToFile.append('Arithmetic weights:')

    print('Pre-trained word2vec model performance:')
    summaryToFile.append('Pre-trained word2vec model performance:')
    createFeatureVectors(shuffledTotalCorpus, 'LR', modelPreTrained)

    print('My word2vec model performance:')
    summaryToFile.append('My word2vec model performance:')
    createFeatureVectors(shuffledTotalCorpus, 'LR', modelSelfTrained)

    print('-------------------------------------------------------------------------------------------------------------------')
    summaryToFile.append('-------------------------------------------------------------------------------------------------------------------')

    print('Random weights:')
    summaryToFile.append('Random weights:')

    print('Pre-trained word2vec model performance:')
    summaryToFile.append('Pre-trained word2vec model performance:')
    createFeatureVectors(shuffledTotalCorpus, 'LR', modelPreTrained, weightMod='random')

    print('My word2vec model performance:')
    summaryToFile.append('My word2vec model performance:')
    createFeatureVectors(shuffledTotalCorpus, 'LR', modelSelfTrained, weightMod='random')

    print('-------------------------------------------------------------------------------------------------------------------')
    summaryToFile.append('-------------------------------------------------------------------------------------------------------------------')

    print('My weights:')
    summaryToFile.append('My weights:')

    print('Pre-trained word2vec model performance:')
    summaryToFile.append('Pre-trained word2vec model performance:')
    createFeatureVectors(shuffledTotalCorpus, 'LR', modelPreTrained, weightMod='special')

    print('My word2vec model performance:')
    summaryToFile.append('My word2vec model performance:')
    createFeatureVectors(shuffledTotalCorpus, 'LR', modelSelfTrained, weightMod='special')

    createSummaryFile(summaryOutputPath)

    print()
    print(datetime.datetime.now())


def createSummaryFile(path):

    f = open(path, 'w+', encoding='utf-8')

    for line in summaryToFile:
        f.write(line + '\n')



@ignore_warnings(category=ConvergenceWarning)
def createFeatureVectors(totalCorpus, classifier, model, featureList=None, vectorType='normal', weightMod=None):

    totalDf = pd.DataFrame.from_dict(totalCorpus)     # create a data frame for the labeled sentences
    y = totalDf['class']     # create a column for of the labels
    X = totalDf['text']

    # print(y.shape)
    # print(X.shape)

    myVectorXtrain = []
    sumOfVec = np.zeros(shape=(300,))


    total_accuracy_score = 0
    total_precision_score = 0
    total_recall_score = 0
    total_f1_score = 0

    if weightMod == 'random':
        # print('Random weight')
        for sentence in X:
            for word in sentence.split():
                if word in model.vocab:
                    vecCalc = model.word_vec(word) * np.random.rand()
                    sumOfVec += vecCalc
            sumOfVec /= len(sentence.split())
            myVectorXtrain.append(sumOfVec.tolist())


    elif weightMod == 'special':
        # print('Special weight')
        for sentence in X:
            length = len(sentence.split())
            counter = length
            forwardFlag = False

            for word in sentence.split():
                if forwardFlag is False and counter > 0:
                    counter -= 2
                    if counter <= 0:
                        forwardFlag = True
                else:
                    counter += 2

                weight = counter / length
                if weight > 1:
                    weight = 1
                elif weight < 0:
                    weight = 0

                if word in model.vocab:
                    vecCalc = model.word_vec(word) * weight
                    sumOfVec += vecCalc
            sumOfVec /= length
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


    myVectorXtrain = np.array(myVectorXtrain)
    # print(myVectorXtrain.shape)

    lr = LogisticRegression(max_iter=100)
    cv_results = cross_validate(lr, myVectorXtrain, y, cv=10, scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'])

    for result in cv_results['test_accuracy']:
        total_accuracy_score += result
    for result in cv_results['test_precision_weighted']:
        total_precision_score += result
    for result in cv_results['test_recall_weighted']:
        total_recall_score += result
    for result in cv_results['test_f1_weighted']:
        total_f1_score += result

    total_accuracy_score /= 10
    total_precision_score /= 10
    total_recall_score /= 10
    total_f1_score /= 10

    total_accuracy_score = int(total_accuracy_score*10000)/100
    total_precision_score = int(total_precision_score*10000)/100
    total_recall_score = int(total_recall_score*10000)/100
    total_f1_score = int(total_f1_score*10000)/100

    print('Accuracy: ', total_accuracy_score)
    summary = str('Accuracy: ' + str(total_accuracy_score))
    summaryToFile.append(summary)

    print('Precision: ', total_precision_score)
    summary = str('Precision: ' + str(total_precision_score))
    summaryToFile.append(summary)

    print('Recall: ', total_recall_score)
    summary = str('Recall: ' + str(total_recall_score))
    summaryToFile.append(summary)

    print('F1: ', total_f1_score)
    summary = str('F1: ' + str(total_f1_score))
    summaryToFile.append(summary)



def readAndLabel(directory):

    label = 0
    totalCorpus = []
    fileCorpus = []


    for currentFile in os.listdir(directory):
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


def phaseA(pathPreTrained, pathSelfTrained):

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