import os
from typing import List
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


labels = ['septic', 'aseptic']


def extract_clf_metrics(y_true: List[str], y_predicted: List[str], output_path: str, model_name: str):


    print(f"Model: {model_name}")

    # Confusion matrix
    cm = confusion_matrix(y_true=y_true, y_pred=y_predicted, labels=labels)
    cf_plot = sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=labels, yticklabels=labels)
    cf_plot.set_xlabel("Predicted")
    cf_plot.set_ylabel("True")
    plt.savefig(os.path.join(output_path, model_name, "cm.png"))

    
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


    print("-------------------------------------------------------------------------------")


def main():

    models = ["squeezenet", "googlenet", "resnet18", "resnet50", "darknet19"]

    for model in models:

        results_path = os.path.join(os.getcwd(), "data", "results", "squeezenet")

        y_true_df = pd.read_csv(os.path.join(results_path, "squeezenet_ytrue.csv"), header=None)
        y_true = y_true_df.loc[:,0].to_list()
        y_pred_df = pd.read_csv(os.path.join(results_path, "squeezenet_ypred.csv"), header=None)
        y_pred = y_pred_df.loc[:,0].to_list()

        extract_clf_metrics(y_true, y_pred, results_path, "squeezenet")



    








if __name__ == "__main__":
    main()

