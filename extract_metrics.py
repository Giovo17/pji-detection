import os
from typing import List, Dict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

labels = ['septic', 'aseptic']
label2id = {'aseptic': 0, 'septic': 1}
id2label = {0: 'aseptic', 1: 'septic'}


def extract_clf_metrics(y_true: List[str], y_pred: List[str], output_path: str, model_name: str) -> Dict[str, float]:
    """Extract classification metrics."""

    # Confusion matrix
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    cf_plot = sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=labels, yticklabels=labels)
    cf_plot.set_xlabel("Predicted")
    cf_plot.set_ylabel("True")
    plt.savefig(os.path.join(output_path, model_name + "_cm.png"))

    # Accuracy, precision, recall, F1-score
    y_true = list(map(label2id.get, map(str, y_true)))
    y_pred = list(map(label2id.get, map(str, y_pred)))

    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)

    return {"Classifier": model_name,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4)}



def main():

    models = ["squeezenet", "googlenet", "resnet18", "resnet50", "darknet19"]
    labels_mapping = {"Pazienti_Settici": "septic", 
                      "Pazienti_Asettici": "aseptic"}

    results = pd.DataFrame(columns=["Classifier", "Accuracy", "Precision", "Recall", "F1-Score"])

    for model in models:
        results_path = os.path.join(os.getcwd(), "data", "results", model)

        y_true_df = pd.read_csv(os.path.join(results_path, model + "_ytrue.csv"), header=None)
        y_true = y_true_df.loc[:,0].to_list()
        y_true = list(map(labels_mapping.get, map(str, y_true)))
        y_pred_df = pd.read_csv(os.path.join(results_path, model + "_ypred.csv"), header=None)
        y_pred = y_pred_df.loc[:,0].to_list()
        y_pred = list(map(labels_mapping.get, map(str, y_pred)))
        

        metrics_dict = extract_clf_metrics(y_true, y_pred, results_path, model)

        results = pd.concat([results, pd.DataFrame(metrics_dict, index=[0])])

    print(results)
    results.to_csv(os.path.join(os.getcwd(), "data", "results", "results.csv"), index=False)


if __name__ == "__main__":
    main()

