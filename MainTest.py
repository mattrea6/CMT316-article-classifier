# Matthew Rea
# c1737407
# This is the main file for running the machine learning routine.

import GetData, Preprocess, FeatureSelection, Visualise
import sklearn

# constants
dataSplits = [80,10,10]
articleTypes = ["business", "entertainment", "politics", "sport", "tech"]

# this will test the machine learning with certain parameters
# trained using the training partition and tested with the development partition
def train_dev_classifier(rawTrainData, rawDevData, ngrams, noFeatures):
    print("\nStarting test with {} features, {}-grams".format(noFeatures, ngrams))
    # tokenise and lemmatize training data
    print("Tokenising training data...")
    processedTrainData = Preprocess.tokenise_lemmatise(rawTrainData, ngrams)
    # get unique unigram list and word frequency vector
    wordList, wordFrequencyVector = FeatureSelection.get_word_list_vector(processedTrainData, noFeatures)
    # vectorise test set
    print("\nVectorising training data...")
    trainData, trainLabels = Preprocess.vectorise_dataset(processedTrainData, wordList)
    # train classifier
    print("\nTraining classifier...")
    classifier = sklearn.svm.SVC(kernel = "linear", gamma = 'auto')
    classifier.fit(trainData, trainLabels)
    print("Classifier trained.")
    # tokenise and lemmatise development data
    print("Tokenising development data...")
    processedDevData = Preprocess.tokenise_lemmatise(rawDevData, ngrams)
    # vectorise train set
    print("\nVectorising development data...")
    devData, devLabels = Preprocess.vectorise_dataset(processedDevData, wordList)
    # classify the test data and print the report.
    print("\nTesting classifier...")
    results = classifier.predict(devData)
    print("Test complete.\nResults: ")
    result = sklearn.metrics.classification_report(devLabels, results, output_dict = True)

    with open("results\\n={}, f={}.txt".format(ngrams, noFeatures), "w+") as outFile:
        resultStr = sklearn.metrics.classification_report(devLabels, results)
        outFile.write(resultStr)

    result['ngrams'] = ngrams
    result['noFeatures'] = noFeatures
    result['classifier'] = classifier
    result['wordList'] = wordList
    Visualise.make_stacked_bar(devLabels, results, "n={}, f={}".format(ngrams, noFeatures))
    return result

# first get the data and split it into training, testing and development sets
rawData = GetData.get_all_files()
train, test, dev = Preprocess.split_data(rawData, dataSplits)

testResults = []

for ngrams in [1,2,3]:
    for noFeatures in [500, 1000, 2000, 4000]:
        testResults.append(train_dev_classifier(train, dev, ngrams, noFeatures))

bestF1 = 0
for result in testResults:
    if result['weighted avg']['f1-score'] > bestF1:
        bestResult = result
        bestF1 = result['macro avg']['f1-score']

print("Best classifier used {} features, {}-grams".format(bestResult['noFeatures'], bestResult['ngrams']))
print("Best classifier got a macro-average F1 score of {} on development set.".format(bestF1))
print("Testing best classifier against test set...")
# now take the best classifier found and test using the test data.
# tokenise and lemmatise test data
processedTestData = Preprocess.tokenise_lemmatise(test, bestResult['ngrams'])
# vectorise train set
print("\nVectorising test data...")
testData, testLabels = Preprocess.vectorise_dataset(processedTestData, bestResult['wordList'])
# classify the test data, print the report and make the graph.
print("\nTesting classifier...")
results = bestResult['classifier'].predict(testData)
resultStr = sklearn.metrics.classification_report(testLabels, results)
print(resultStr)
with open("results\\test n={}, f={}.txt".format(bestResult['ngrams'], bestResult['noFeatures']), "w+") as outFile:
    outFile.write(resultStr)
Visualise.make_stacked_bar(testLabels, results, "test n={}, f={}".format(bestResult['ngrams'], bestResult['noFeatures']))
