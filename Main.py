# Matthew Rea
# c1737407
# This is the main file for running the machine learning routine.

import GetData, Preprocess, FeatureSelection, Visualise
import sklearn

# constants
dataSplits = [80,20]
articleTypes = ["business", "entertainment", "politics", "sport", "tech"]

# get data from file
rawData = GetData.get_all_files()
# split data into train and test
rawTrain, rawTest = Preprocess.split_data(rawData, dataSplits)
# tokenise and lemmatize train data
processedTrainData = Preprocess.tokenise_lemmatise(rawTrain, 1)
# get unique unigram list and word frequency vector
wordList, wordFrequencyVector = FeatureSelection.get_word_list_vector(processedTrainData, 4000)
# vectorise train data
print("\nVectorising training data...")
trainData, trainLabels = Preprocess.vectorise_dataset(processedTrainData, wordList)
# train classifier
print("\nTraining classifier...")
classifier = sklearn.svm.SVC(kernel = "linear", gamma = 'auto')
classifier.fit(trainData, trainLabels)
print("Classifier trained.")
# tokenise and lemmatize test data
processedTestData = Preprocess.tokenise_lemmatise(rawTest, 1)
# vectorise test set
print("\nVectorising training data...")
testData, testLabels = Preprocess.vectorise_dataset(processedTestData, wordList)
# classify the test data and print the report.
print("\nTesting classifier...")
results = classifier.predict(testData)
print("Test complete.\n")
print(sklearn.metrics.classification_report(testLabels, results))
Visualise.make_stacked_bar(testLabels, results)
