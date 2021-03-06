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
    c_ax.set_title('AUC curve on Validation Set')
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    
## what other performance statistics do you want to include here besides AUC?     

# function to plot the precision_recall_curve. You can utilizat precision_recall_curve imported above
def plot_precision_recall_f1_curve(c_axs, t_y, p_y):
    
    # precision recall curve
    precision, recall, thresholds = precision_recall_curve(t_y, p_y)
    c_axs[0].plot(recall, precision, label = '%s (AP Score:%0.2f)'  % ('Pneumonia', average_precision_score(t_y,p_y)))
    c_axs[0].legend()
    c_axs[0].set_title('Precision-Recall Curve on Validation Set')
    c_axs[0].set_xlabel('Recall')
    c_axs[0].set_ylabel('Precision')
    
    # precision, recall, F1, thresholds
    precision = np.asarray(precision[:-1])
    recall = np.asarray(recall[:-1])
    f1 = calc_f1(precision, recall)
    maxf1index = np.nanargmax(f1)
    c_axs[1].plot(thresholds, precision, label = 'precision')
    c_axs[1].plot(thresholds, recall, label = 'recall')
    c_axs[1].plot(thresholds, f1, label = 'F1-score')
    c_axs[1].legend()
    #c_axs[1].set_title()
    c_axs[1].set_xlabel('Threshold')
    c_axs[1].set_ylabel('Score')  
    summary_text = f'With max F1-score={f1[maxf1index]:.3}, threshold={thresholds[maxf1index]:.3}, Recall={recall[maxf1index]:.3}, Precision = {precision[maxf1index]:.2} on Validation Set'
    return(summary_text)



#Also consider plotting the history of your model training:

def plot_history(history_path):
    
    history = np.load(history_path,allow_pickle='TRUE').item() 
    number_of_axs = len(history)
    colnum = 3
    rownum = (number_of_axs - 1)//colnum + 1
    fig, axs = plt.subplots(rownum, colnum, figsize = (15, rownum*5))
    axs = axs.flatten()
    for ax in axs[number_of_axs:]:
        ax.remove()
    
    N = len(history["loss"])
    keys = list(history.keys())
    keys.remove('loss')
    keys = ['loss'] + keys
    for i, key in enumerate(keys):
        axs[i].plot(np.arange(0, N), history[key], label=key)
        if key == "loss":
            mode = "Training"
        else:
            mode = 'Validation'
        axs[i].set_title(f"{mode} {key} on Dataset")
        axs[i].set_xlabel("Epoch #")
        axs[i].set_ylabel("Loss/Accuracy")
        axs[i].legend(loc="lower left")
        handles, labels = axs[i].get_legend_handles_labels()
        axs[i].legend(handles, labels, loc =  1)



def plot_performance(prediction_path, groundtruth_path, title = ''):
    # Get prediction
    probability = np.load(prediction_path)
    ground_truth = np.load(groundtruth_path)
    
    
    # plot results 
    number_of_axs = 3 
    colnum = 3
    rownum = (number_of_axs - 1)//colnum + 1 
    fig, axs = plt.subplots(rownum, colnum, figsize = (15, rownum*5))
    axs = axs.flatten()
    for ax in axs[number_of_axs:]:
        ax.remove()
    plot_auc(axs[0], ground_truth, probability)
    summary_text = plot_precision_recall_f1_curve(axs[1:], ground_truth, probability)
    fig.suptitle(f'''{title} ({summary_text})''')
    plt.show()
    
        
# function to calculate the F1 score
def  calc_f1(prec,recall):
    return 2.0*(prec*recall)/(prec+recall)        