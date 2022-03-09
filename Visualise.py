# Matthew Rea
# c1737407
# this file contains a function that creates a stacked bar chart for the data.
# code (heavily) adapted from https://gist.github.com/nils-fl/d0cafd089dc1e6204a08a7f8d617f0dc#file-stacked_mosaic_plot_classification-ipynb

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# creates a stacked bar chart for classification.
def make_stacked_bar(realLabels, predLabels, filename="Test"):
    labels = ["business", "entertainment", "politics", "sport", "tech"]
    # set up the dictionary with real-predicted label pairs
    results = {}
    for i in labels:
        for j in labels:
            results["{}-{}".format(i, j)] = 0

    # count all of the matches and non matches in the result set
    for i, pred in enumerate(predLabels):
        # looks at what the real label and predicted label are
        result = "{}-{}".format(realLabels[i], pred)
        # increments this category in the dictionary
        results[result] += 1

    # now turn these into separate lists so pyplot can use them
    bar1 = []
    bar2 = []
    bar3 = []
    bar4 = []
    bar5 = []
    # this assigns all of the correct results to the correct places in each list for displaying
    for i, label in enumerate(labels):
        bar1.append(results["{}-{}".format(label, label)])
        bar2.append(results["{}-{}".format(label, labels[(i+1)%len(labels)])])
        bar3.append(results["{}-{}".format(label, labels[(i+2)%len(labels)])])
        bar4.append(results["{}-{}".format(label, labels[(i+3)%len(labels)])])
        bar5.append(results["{}-{}".format(label, labels[(i+4)%len(labels)])])

    # set up bottom points for each set of results so bars sit in the correct place
    bt1 = [bar1[i]+bar2[i] for i in range(len(labels))]
    bt2 = [bt1[i]+bar3[i] for i in range(len(labels))]
    bt3 = [bt2[i]+bar4[i] for i in range(len(labels))]

    # set up colours so each category has a consistent colour
    # business = blue, ent = orange, pol = green etc
    colours = ["blue", "orange", "green", "red", "purple"]
    colour1 = []
    colour2 = []
    colour3 = []
    colour4 = []
    colour5 = []
    # set the correct colours for the bars in the same way as before for results
    for i, colour in enumerate(colours):
        colour1.append(colours[(i)%len(colours)])
        colour2.append(colours[(i+1)%len(colours)])
        colour3.append(colours[(i+2)%len(colours)])
        colour4.append(colours[(i+3)%len(colours)])
        colour5.append(colours[(i+4)%len(colours)])

    width = 0.35
    fig, ax = plt.subplots()
    # set each bar and stack on top of one another
    ax.bar(labels, bar1, width, color = colour1)
    ax.bar(labels, bar2, width, bottom = bar1, color = colour2)
    ax.bar(labels, bar3, width, bottom = bt1, color = colour3)
    ax.bar(labels, bar4, width, bottom = bt2, color = colour4)
    ax.bar(labels, bar5, width, bottom = bt3, color = colour5)
    # create extra handles for the colours in the legend.
    legendHandles = [mpatches.Patch(color=colours[i], label=labels[i]) for i in range(5)]
    ax.legend(handles=legendHandles)
    # set graph parameters
    ax.set_xlabel('Actual Labels')
    ax.set_ylabel('Predicted Labels')
    ax.set_title('Accuracy of classification of all labels - {}'.format(filename))
    if filename != "Test":
        plt.savefig("results\\{}.png".format(filename))
    else:
        plt.show()
    return None
