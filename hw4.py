import os

import gensim
from gensim.models import Word2Vec
import gc
from gensim.models import KeyedVectors


path = r'C:\Users\oron.werner\PycharmProjects\NLP\hw4Inputs\allCountryFiles'
inputAllCountries = r'C:\Users\oron.werner\PycharmProjects\NLP\hw4Inputs\allCountryFiles\SumOfAll.txt'
hw4Inputs = r'C:\Users\oron.werner\PycharmProjects\NLP\hw4Inputs'
pathSelfTrained = r'C:\Users\oron.werner\PycharmProjects\NLP\hw4Inputs\self_trained_model.vec'
pathPreTrained = r'C:\Users\oron.werner\PycharmProjects\NLP\hw4Inputs\wiki.en.100k.vec'
allUsersOfCountry = r'C:\Users\oron.werner\PycharmProjects\NLP\hw4Inputs\allUsersOfCountryPhaseB'


def main():

    combineSentences(allUsersOfCountry)



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

            newPath = folderPath + '\\chunck\\' + 'combined20' + currentFile

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