Matthew Rea
c1737407

The steps for this classification are separated into
  GetData.py
  Preprocess.py
  FeatureSelection.py
  Visualise.py

To begin, the data directory 'bbc' must be in the same directory as 'Main.py'

To run the test to select and determine the best parameters use command;
> python MainTest.py
To run this test, there must be a 'results' directory in the same directory as 'MainTest.py'
This will create 13 .png files and 13.txt files in this directory.

To run the classification using the pre-selected best parameters, use command;
> python Main.py
Results are printed to the terminal.

GetData.py contains functions that get raw data from all text files under bbc\
Preprocess.py contains functions that tokenise, lemmatise, vectorise and convert articles to n-gram form.
FeatureSelection.py contains functions that count word frequency and create word frequency vectors
Visualise.py contains a function that takes classification results and visualises them.
