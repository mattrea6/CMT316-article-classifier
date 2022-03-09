# Matthew Rea
# c1737407
# this file contains functions that extract all raw text from files

import os

# get raw text from file
def get_raw_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as inFile:
            return inFile.readlines()
    else:
        print("File {} does not exist.".format(filepath))
        return [] # fail with empty list

# get all files in one directory
def get_all_files_directory(directory):
    if os.path.exists(directory):
        allFiles = []
        # list all files in directory and grab them one at a time
        fileNames = os.listdir(directory)
        for fileName in fileNames:
            allFiles.append(get_raw_file(directory+"\\"+fileName))
        print("Retrieved {} files from {}.".format(len(allFiles), (directory+"\\")))
        return allFiles
    else:
        print("Directory {} does not exist.".format(directory))
        return [] # fail with empty list

# gets all files for all directories
# return data as dictionary of lists
def get_all_files():
    print("Retrieving all files from bbc\\...")
    root = "bbc\\{}"
    articleTypes = ["business", "entertainment", "politics", "sport", "tech"]
    data = {}
    for articleType in articleTypes:
        data[articleType] = get_all_files_directory(root.format(articleType))
    return data
