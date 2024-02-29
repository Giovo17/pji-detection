import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


labels = ['septic', 'aseptic']


def extract_clf_metrics(df: pd.DataFrame):

    #y_true =
    #y_predicted = 

    # Confusion matrix
    cm = confusion_matrix(y_true=y_true, y_pred=y_predicted, labels=labels)
    cf_plot = sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=labels, yticklabels=labels)
    cf_plot.set_xlabel("Predicted")
    cf_plot.set_ylabel("True")
    #cf_plot.save

    
    # Calculate average accuracy and overall f1-score
    accuracy = accuracy_score(y_true, y_predicted)

    f1_micro = f1_score(y_true, y_predicted, average='micro')
    f1_macro = f1_score(y_true, y_predicted, average='macro')
    f1_weighted = f1_score(y_true, y_predicted, average='weighted')

    print("Accuracy: " + str(round(accuracy, 4)))
    print("F1 score averaged micro: " + str(round(f1_micro, 4)))
    print("F1 score averaged macro: " + str(round(f1_macro, 4)))
    print("F1 score averaged weighted: " + str(round(f1_weighted, 4)))



    # Metrics of septic class
    recall_class_0 = recall_score(y_true, y_predicted, labels=['septic'])
    precision_class_0 = precision_score(y_true, y_predicted, labels=['septic'])
    f1_class_0 = f1_score(y_true, y_predicted, labels=['septic'])

    print("Recall for class septic: " + str(round(recall_class_0, 4)))
    print("Precision for class septic: " + str(round(precision_class_0, 4)))
    print("F1-score for class septic: " + str(round(f1_class_0, 4)))


    # Metrics of aseptic class
    recall_class_1 = recall_score(y_true, y_predicted, labels=['aseptic'])
    precision_class_1 = precision_score(y_true, y_predicted, labels=['aseptic'])
    f1_class_1 = f1_score(y_true, y_predicted, labels=['aseptic'])

    print("Recall for class aseptic: " + str(round(recall_class_1, 4)))
    print("Precision for class aseptic: " + str(round(precision_class_1, 4)))
    print("F1-score for class aseptic: " + str(round(f1_class_1, 4)))





def main():
    extract_clf_metris()


if __name__ == "__main__":
    main()

