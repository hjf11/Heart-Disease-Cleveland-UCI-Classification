import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from itertools import product
from random import sample

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm

def print_classification_report_from_model(model, X, y):
    if isinstance(model, keras.Model):
        probs = model.predict(X)
        pred = (probs > 0.5).astype(int)
    else:
        pred = model.predict(X)
    
    print("Classification Report (Train data):")
    print(classification_report(y, pred))
    

def print_classification_report(y, pred):
    print("Classification Report:")
    print(classification_report(y, pred))
    
    
def print_confusion_matrix(y, pred, title, filename, folder='images', labels=['No Disease', 'Disease']):    
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
    plt.title('Confusion Matrix for ' + title)
    
    plt.tight_layout()
    
    full_path = os.path.join(folder, filename)
    plt.savefig(full_path, dpi=300)
    plt.show()
    
    
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
    
    
def print_metrics(model, X, y, title, filename): 
    # check if it is keras model
    if isinstance(model, keras.Model):
        probs = model.predict(X, verbose=0)
        pred = (probs > 0.5).astype(int)
    else:
        pred = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
    print(f'Accuracy: {accuracy_score(y, pred)}')
    print_classification_report(y, pred)
    print()
    print_confusion_matrix(y, pred, title, filename)
    print_roc_curve(y, probs)
    
    
def plot_roc_curves(models, model_names, X, y, filename='roc_curves.png', folder='images'):
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
        if isinstance(model, keras.Model):
            probs = model.predict(X, verbose=0)
        else:
            probs = model.predict_proba(X)[:, 1]

        fpr, tpr, _ = roc_curve(y, probs)
        auc = roc_auc_score(y, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", linewidth=2)

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    full_path = os.path.join(folder, filename)
    plt.savefig(full_path, dpi=300)
    plt.show()