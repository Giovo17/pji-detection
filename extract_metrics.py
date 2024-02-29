import os
from typing import List
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


labels = ['septic', 'aseptic']


def extract_clf_metrics(y_true: List[str], y_pred: List[str], output_path: str, model_name: str):
    """Extract classification metrics."""

    print(f"Model: {model_name}")

    # Confusion matrix
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    cf_plot = sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=labels, yticklabels=labels)
    cf_plot.set_xlabel("Predicted")
    cf_plot.set_ylabel("True")
    plt.savefig(os.path.join(output_path, model_name + "_cm.png"))


    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)

    print("Accuracy: " + str(round(accuracy, 4)))
    print("Precision: " + str(round(precision, 4)))
    print("Recall: " + str(round(recall, 4)))
    print("F1-score: " + str(round(f1, 4)))


    print("-------------------------------------------------------------------------------")


def main():

    models = ["squeezenet", "googlenet", "resnet18", "resnet50", "darknet19"]
    labels_mapping = {"Pazienti_Settici": "septic", 
                      "Pazienti_Asettici": "aseptic"}

    for model in models:

        results_path = os.path.join(os.getcwd(), "data", "results", model)

        y_true_df = pd.read_csv(os.path.join(results_path, model + "_ytrue.csv"), header=None)
        y_true = y_true_df.loc[:,0].to_list()
        y_true = list(map(labels_mapping.get, map(str, y_true)))
        y_pred_df = pd.read_csv(os.path.join(results_path, model + "_ypred.csv"), header=None)
        y_pred = y_pred_df.loc[:,0].to_list()
        y_pred = list(map(labels_mapping.get, map(str, y_pred)))
        

        extract_clf_metrics(y_true, y_pred, results_path, model)




if __name__ == "__main__":
    main()

