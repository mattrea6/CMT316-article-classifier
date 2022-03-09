# Matthew Rea
# c1737407
# this file contains functions that preprocess the data in various ways.

import random
import numpy as np
import nltk

# make sure packages are up to date
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# takes article from list of sentences to list of tokens
def tokenise_article(data):
    tokenData = []
    for sentence in data:
        if len(sentence) > 1: # this strips out the empty lines
            tokenData.extend(nltk.tokenize.word_tokenize(sentence))
    return tokenData

# takes data from tokeniser and lemmatises
def lemmatize_tokens(data):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemma_data = []
    for token in data:
        # lemmatise and also send to lowercase
        lemma_data.append(lemmatizer.lemmatize(token).lower())
    return lemma_data

# words or symbols that stopwords doesn't include but I want
extraStopwords = [".",",","-","--","``","'","'s",'"',"''", ")", "(", ":"]
# takes the stopwords out of the data
def remove_stopwords(data):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    for char in extraStopwords:
        stopwords.add(char)
    # returns article with no stopwords
    return [word for word in data if word not in stopwords]

# converts the data from unigrams to bigrams
def make_bigram(data):
    newData = {}
    for articleType in data.keys():
        newData[articleType] = []
        for article in data[articleType]:
            bigrams = []
            for i in range(len(article)-1):
                # just take the word and add the word after it too
                bigrams.append("{} {}".format(article[i], article[i+1]))
            newData[articleType].append(bigrams)
    return newData

# converts the data from unigrams to trigrams
def make_trigram(data):
    newData = {}
    for articleType in data.keys():
        newData[articleType] = []
        for article in data[articleType]:
            trigrams = []
            for i in range(len(article)-2):
                # just take the word and add the two words after it too
                trigrams.append("{} {} {}".format(article[i], article[i+1], article[i+2]))
            newData[articleType].append(trigrams)
    return newData

# function to split data into different sized sets
# should work for any proportions, any number of splits
# this method is destructive to the original data
def split_data(data, in_ratio):
    # ratio of split sizes must equal 100
    if sum(in_ratio) != 100:
        print("Split ratio {} != 100".format(sum(in_ratio)))
        return []
    ratio = [x/100 for x in in_ratio]
    print("\nSplitting data {}".format(ratio))

    # make a list of dictionaries that hold the different splits
    splitData = [{} for i in range(len(ratio))]
    articleTypes = ["business", "entertainment", "politics", "sport", "tech"]
    for article in articleTypes:
        # find the size of the full data and each split size
        setSize = len(data[article])
        splitSizes = [int(setSize*x) for x in ratio]
        # this ensures the sizes are equal
        splitSizes[0] += setSize - sum(splitSizes)
        # this will pop random articles from the original data
        # and place them in different splits until the original data is empty
        for split, splitSize in enumerate(splitSizes):
            splitData[split][article] = []
            for i in range(splitSize):
                curSize = len(data[article])
                # pops a random article from original data and places it in new split
                splitData[split][article].append(data[article].pop(random.randrange(0,curSize)))
        actualRatio = [len(splitData[i][article])/setSize for i in range(len(ratio))]
        #print("Actual split ratio for {}: {}".format(article, actualRatio))
    return splitData

# takes the data from GetData in dictionary form
# completes preprocessing steps and returns as dict
def tokenise_lemmatise(data, gram = 1):
    print("\nTokenising data...")
    processedData = {}
    articleTypes = data.keys()
    # for each type of article
    for articleType in articleTypes:
        print("Tokenising {}...".format(articleType))
        processedData[articleType] = []
        # for each article in list of articles
        for article in data[articleType]:
            # tokenise
            tokens = tokenise_article(article)
            # lemmatize
            lemmas = lemmatize_tokens(tokens)
            # remove stopwords
            stopwordsRemoved = remove_stopwords(lemmas)
            processedData[articleType].append(stopwordsRemoved)
    # now we have a dictionary of articles in tokenised-lemmatized form
    print("Data tokenised.")
    if gram == 2:
        return make_bigram(processedData)
    elif gram == 3:
        return make_trigram(processedData)
    return processedData

# takes article and vocab and turns it into a vector
def vectorise_article(article, vocabulary):
    articleVector = np.zeros(len(vocabulary))
    # go through vocab list
    for i, word in enumerate(vocabulary):
        # set article vector index to count of word in article
        if word in article:
            articleVector[i] = article.count(word)
    return articleVector

# takes the whole dataset and turns it into two lists, ready for training.
def vectorise_dataset(data, vocabulary):
    #print("Vectorising data...")
    dataVectors = []
    dataLabels = []
    for articleType in data.keys():
        print("Vectorising {}...".format(articleType))
        for article in data[articleType]:
            # for each article add its vector form to data vectors
            dataVectors.append(vectorise_article(article, vocabulary))
            # and add a corresponding label to the labels
            dataLabels.append(articleType)
    print("Data vectorised.")
    return dataVectors, dataLabels
