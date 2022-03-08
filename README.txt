Matthew Rea
c1737407

Dependencies;
  os
  operator
  random
  scikit-learn==1.0.2
  nltk==3.7
  numpy==1.22.2
  matplotlib==3.5.1

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
