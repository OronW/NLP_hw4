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
from sklearn.model_selection import KFold
from sklearn import metrics


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
    createFeatureVectors(totalCorpus, 'LR', model, vectorType='manual')
    # if 'and' in model.vocab:
    #     print('HURRAY!')
    #
    # if 'etgf' in model.vocab:
    #     print('haa!')
    # print(model.word_vec('and')[:2])
    # print(model.get_vector('and')[:2])


@ignore_warnings(category=ConvergenceWarning)
def createFeatureVectors(totalCorpus, classifier, model, featureList=None, vectorType='normal'):

    totalDf = pd.DataFrame.from_dict(totalCorpus)     # create a data frame for the labeled sentences
    y = totalDf['class']     # create a column for of the labels
    X = totalDf['text']

    sum = 0

    weight = 1

    kf = KFold(n_splits=10, random_state=1, shuffle=True)

    for train_index, test_index in kf.split(X):
        # print("\nTRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


        if vectorType == 'manual':
            myVectorXtrain = []
            myVectorXtest = []
            sentenceVector = []
            sumOfVec = np.zeros(shape=(300,))

            for sentence in X_train[:1]:
                print(sentence)
                for word in sentence.split():
                    print(word)
                    if word in model.vocab:
                        vecCalc = model.word_vec(word) * weight
                        print(model.word_vec(word)[0])

                        vecCalc /= len(sentence.split())
                        print(len(sentence.split()))
                        print('vecCalc: ', vecCalc[0])
                        sumOfVec += vecCalc
                        print(sumOfVec[0])
                myVectorXtrain.append(sumOfVec)

            for sentence in X_test:
                st = sentence.split()
                myVectorXtest.append([len(sentence), len(st), sentence.count('!'), sentence.count('?'), sentence.count('.'), sentence.count('\''), sentence.count('I am'), sentence.count('you \' re'), sentence.count('. . .'), sentence.count('I \' m')])


            X_train_dtm = np.array(myVectorXtrain)  # create document - term matrix for the words
            X_test_dtm = np.array(myVectorXtest)



        # -- this part for LR classifier --
        elif classifier == 'LR':
            lr = LogisticRegression(max_iter=500)
            lr.fit(X_train_dtm, y_train)
            y_pred_class = lr.predict(X_test_dtm)
            sum += metrics.accuracy_score(y_test, y_pred_class)

        else:
            print('NO CLASSIFIER SELECTED FOR \'createFeatureVectors\' FUNCTION. ENDING RUN! ')
            exit()


        # print('\nAccuracy score: ', metrics.accuracy_score(y_test, y_pred_class))
        # print('Test sentences by classes:')
        # print(y_test.value_counts())
        # print('Confusion matrix:\n', metrics.confusion_matrix(y_test, y_pred_class))

    # print('\n**********************************')

    acc = sum/10
    total = int(sum*1000)/100

    if classifier == 'NB':
        print('Naïve Bayes:', total)
        summary = str('Naïve Bayes: ' + str(total))
        # summaryToFile.append(summary)

    if classifier == 'LR':
        print('Logistic Regression: ', total)
        summary = str('Logistic Regression: ' + str(total))
        # summaryToFile.append(summary)



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