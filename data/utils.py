import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
import tensorflow.keras as keras

def print_classification_report(y, pred):
    print("Classification Report:")
    print(classification_report(y, pred))
    
    
def print_confusion_matrix(y, pred, labels=['No Disease', 'Disease']):    
    cm = confusion_matrix(y, pred)
    
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(cm, annot=False, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels, cbar=False)
    
    for i in range(2):  # 2 classes (binary classification)
        for j in range(2):
            count = cm[i, j]
            
            max_value = np.max(cm)
            color_intensity = count / max_value  # Values between 0 and 1
            
            # Choose text color based on intensity
            text_color = 'white' if color_intensity > 0.5 else 'black'
            
            # Create the label with count and type
            if i == 0 and j == 0:
                label = f'True Negative\n{count}'
            elif i == 0 and j == 1:
                label = f'False Positive\n{count}'
            elif i == 1 and j == 0:
                label = f'False Negative\n{count}'
            elif i == 1 and j == 1:
                label = f'True Positive\n{count}'
            
            # Annotate the cell with the label (count + classification type)
            plt.text(j + 0.5, i + 0.5, label, ha='center', va='center', color=text_color, fontsize=12)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    plt.show()
    
    # cm = confusion_matrix(y, pred)

    # plt.figure(figsize=(6, 5))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.title('Confusion Matrix')
    # plt.tight_layout()
    # plt.show()
    
    
def print_roc_curve(y, probs):
    fpr, tpr, thresholds = roc_curve(y, probs)

    auc = roc_auc_score(y, probs)

    plt.figure(figsize=(8, 6))
    sns.set_style('whitegrid')
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (random classifier)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    
    
def print_metrics(model, X, y): 
    # check if it is keras model
    if isinstance(model, keras.Model):
        probs = model.predict(X)
        pred = (probs > 0.5).astype(int)
    else:
        pred = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
    
    print_classification_report(y, pred)
    print()
    print_confusion_matrix(y, pred)
    print_roc_curve(y, probs)
    
    
def plot_roc_curves(models, model_names, X, y):
    """
    Plots ROC curves for multiple models.
    
    Args:
        models (list): List of fitted models (must support predict_proba).
        model_names (list): List of model names (strings) corresponding to models.
        X (array): Features.
        y (array): Labels.
    """
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")

    for model, name in zip(models, model_names):
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            # For models like SVM with no predict_proba unless explicitly enabled
            probs = model.decision_function(X)
        else:
            raise ValueError(f"Model '{name}' must support predict_proba or decision_function.")

        fpr, tpr, _ = roc_curve(y, probs)
        auc = roc_auc_score(y, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", linewidth=2)

    # Diagonal random line
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()