import os

path = r'C:\Users\oron.werner\PycharmProjects\NLP\hw4Inputs\allCountryFiles'


def main():

    combineFilesIntoOne(path)


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