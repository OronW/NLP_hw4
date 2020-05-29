import os

path = r'C:\Users\oron.werner\PycharmProjects\NLP\hw4Inputs\allCountryFiles'
inputAllCountries = r'C:\Users\oron.werner\PycharmProjects\NLP\hw4Inputs\allCountryFiles\SumOfAll.txt'

def main():

    sentencesList = sentencesToListOfLists(inputAllCountries)
    print(len(sentencesList))



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