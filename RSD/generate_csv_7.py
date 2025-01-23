from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import json
import os
import csv
def binary_train():
    benchmark = 'test'
    exp_name = ["full"]
    
    
    tmp_design_smell = [
        "Cyclic Hierarchy",
        "Broken Hierarchy",
        "Cyclically-dependent Modularization",  #0.72
        "Deep Hierarchy",
        "Wide Hierarchy",
        "Feature Envy",
        "Multipath Hierarchy",
        "Rebellious Hierarchy",
    ]
    latex = []
    for exp in exp_name:
        metrics = []
        path = os.path.join("./RSD/model", exp)
        for smell in tmp_design_smell:
            latex.append([])
            smell_path = os.path.join(path, f"binary_classification_{smell}_{benchmark}","test_file.json")
            data = json.load(open(smell_path,'r'))
            y_true_label = data['labels']
            y_pred_label = data['preds']
            acc = accuracy_score(y_true_label, y_pred_label)
            recall = recall_score(y_true_label, y_pred_label)
            f1 = f1_score(y_true_label, y_pred_label)
            latex[-1].append(acc)
            latex[-1].append(f1)
            latex[-1].append(recall)
            # latex[-1] = f1
            metrics.append({
                'smell':smell,
                'F1': f1,
                'Recall': recall,
                'Accuracy': acc
            })

        # Calculate means
        n = len(tmp_design_smell)
        mean_f1 = sum(metric['F1'] for metric in metrics) / n
        mean_recall = sum(metric['Recall'] for metric in metrics) / n
        mean_accuracy = sum(metric['Accuracy'] for metric in metrics) / n
        latex.append([])
        latex[-1].append(mean_accuracy)
        latex[-1].append(mean_f1)
        latex[-1].append(mean_recall)
        latex[-1] = mean_f1
        with open(f"./{exp}.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Smell', 'F1', 'Recall', 'Accuracy'])

            for i, metric in enumerate(metrics):
                writer.writerow([metric['smell'], metric['F1'], metric['Recall'], metric['Accuracy']])

            # writer.writerow([])
            writer.writerow(['Mean', mean_f1, mean_recall, mean_accuracy])
        # print(latex)
        

def main():
    
    binary_train()
    
        

if __name__ == "__main__":
    main()