# Matthew Rea
# c1737407
# this file contains functions that perform feature selection.

from operator import itemgetter
import numpy as np

# takes list of articles tokenised and lemmatized and creates a word frequency vector
def count_word_frequency(data):
    wordFrequency = {}
    for article in data:
        for word in article:
            # adds word if not already counted, increments if it is
            if word in wordFrequency:
                wordFrequency[word] += 1
            else:
                wordFrequency[word] = 1
    return wordFrequency

# takes multiple frequency dictionaries and combines them
def combine_word_frequency(wordFrequencies):
    totalFrequency = {}
    for freqDict in wordFrequencies:
        for key in freqDict.keys():
            if key not in totalFrequency:
                totalFrequency[key] = freqDict[key]
            else:
                totalFrequency[key] += freqDict[key]
    return totalFrequency

# takes a frequency dictionary and turns it into a list of words and vector
def make_vector(wordFrequencyDict):
    # make list of dictionary keys and empty vector
    wordList = list(wordFrequencyDict.keys())
    frequencyVector = np.zeros(len(wordFrequencyDict))
    for i, word in enumerate(wordList):
        # set each vector index to word count
        frequencyVector[i] = wordFrequencyDict[word]
    return wordList, frequencyVector

# gets n most or least frequent words in dictionary
def select_n_most_frequent(wordFrequencyDict, n, mostFrequent = True):
    return dict(sorted(wordFrequencyDict.items(), key = itemgetter(1), reverse = mostFrequent)[:n])

# takes data in dictionary form and produces a unique word list and a word frequency vector
def get_word_list_vector(data, n):
    wordFrequencies = []
    # get the word frequencies for each article type
    for articleType in data.keys():
        wordFrequencies.append(count_word_frequency(data[articleType]))
    # combine these into one word frequency dictionary
    totalWordFrequency = combine_word_frequency(wordFrequencies)
    # take the n most or least frequent words
    mostFrequent = select_n_most_frequent(totalWordFrequency, n, True)
    # turn the dictionary into a list and a vector
    wordList, vector = make_vector(mostFrequent)
    return wordList, vector
