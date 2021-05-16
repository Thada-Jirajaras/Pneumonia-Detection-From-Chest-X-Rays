import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
import keras.backend as K
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import model_from_json
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, plot_precision_recall_curve, f1_score, confusion_matrix

def predict(model_path, weight_path, val_gen, 
            prediction_path, groundtruth_path):
    model = load_model(model_path, weight_path)
    pred_Y = model.predict(val_gen, verbose = True)
    probability = pred_Y.flatten()   
    np.save(prediction_path, probability)
    np.save(groundtruth_path, np.asarray(val_gen.labels))
    


def load_model(model_path, weight_path):
    with open(model_path, 'r') as json_file:
         model = model_from_json(json_file.read())  
    model.load_weights(weight_path)
    return model

def plot_auc(c_ax, t_y, p_y):
    ## Hint: can use scikit-learn's built in functions here like roc_curve
     # Todo
    fpr, tpr, thresholds = roc_curve(t_y, p_y)
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % ('Pneumonia', auc(fpr, tpr)))
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    
## what other performance statistics do you want to include here besides AUC?     

# function to plot the precision_recall_curve. You can utilizat precision_recall_curve imported above
def plot_precision_recall_curve(c_ax, t_y, p_y):
    precision, recall, thresholds = precision_recall_curve(t_y, p_y)
    c_ax.plot(recall, precision, label = '%s (AP Score:%0.2f)'  % ('Pneumonia', average_precision_score(t_y,p_y)))
    c_ax.legend()
    c_ax.set_xlabel('Recall')
    c_ax.set_ylabel('Precision')



#Also consider plotting the history of your model training:

def plot_history(c_axs, history):
    
    N = len(history["loss"])
    for i, key in enumerate(history.keys()):
        c_axs[i].plot(np.arange(0, N), history[key], label=key)
        c_axs[i].set_title(f"Training {key} on Dataset")
        c_axs[i].set_xlabel("Epoch #")
        c_axs[i].set_ylabel("Loss/Accuracy")
        c_axs[i].legend(loc="lower left")
    
def plot_performance(prediction_path, groundtruth_path, history_path, title = ''):
    # Get prediction
    probability = np.load(prediction_path)
    ground_truth = np.load(groundtruth_path)
    history = np.load(history_path,allow_pickle='TRUE').item()
    
    # plot results 
    
    fig, axs = plt.subplots(1, 2 + len(history), figsize = (15, 6))
    plot_auc(axs[0], ground_truth, probability)
    plot_precision_recall_curve(axs[1], ground_truth, probability)
    plot_history(axs[2:], history) 
    fig.suptitle(title)
    plt.show()
    
        
# function to calculate the F1 score
def  calc_f1(prec,recall):
    return 2*(prec*recall)/(prec+recall)        