#!/usr/bin/env python
# _*_ coding: utf-8 _*_
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import math
import os, re, sys
import pandas as pd
from collections import Counter

# Save forecast results
def save_predict_result(data, output):
    with open(output, 'w') as f:
        if len(data)>1:
            for i in range(len(data)):
                f.write('# result for fold %d\n' % (i + 1))
                for j in range(len(data[i])):
                    f.write('%d\t%s\n' % (data[i][j][0], data[i][j][2]))
        else:
            for i in range(len(data)):
                f.write('# result for predict\n')
                for j in range(len(data[i])):
                    f.write('%d\t%s\n' % (data[i][j][0], data[i][j][2]))
        f.close()
    return None
    
# Plot the ROC curve and return the AUC value
def plot_roc_curve(data, output, label_column=0, score_column=2):
    datasize = len(data)
    tprs = []
    aucs = []
    fprArray = []
    tprArray = []
    thresholdsArray = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(len(data)):
        fpr, tpr, thresholds = roc_curve(data[i][:, label_column], data[i][:, score_column])
        fprArray.append(fpr)
        tprArray.append(tpr)
        thresholdsArray.append(thresholds)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'blueviolet', 'deeppink'])
    fig = plt.figure(figsize=(7,7))
    for i, color in zip(range(len(fprArray)), colors):
        if datasize>1:
            plt.plot(fprArray[i], tprArray[i], lw=1, alpha=0.7, color=color,
                     label='ROC fold %d (AUC = %0.4f)' % (i + 1, aucs[i]))
        else:
            plt.plot(fprArray[i], tprArray[i], lw=1, alpha=0.7, color=color,
                     label='ROC (AUC = %0.4f)' %aucs[i])
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Random', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    # Calculate the standard deviation
    std_auc = np.std(aucs)
    if datasize>1:
        plt.plot(mean_fpr, mean_tpr, color='blue',
                 label=r'Mean ROC (AUC = %0.4f $\pm$ %0.3f)' % (mean_auc, std_auc),
                 lw=2, alpha=.9)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    if datasize>1:
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(output)
    plt.close(0)
    return mean_auc, aucs
    
# Plot the PRC curve and return the PRC value
def plot_prc_curve(data, output, label_column=0, score_column=2):
    datasize = len(data)
    precisions = []
    aucs = []
    recall_array = []
    precision_array = []
    mean_recall = np.linspace(0, 1, 100)

    for i in range(len(data)):
        precision, recall, _ = precision_recall_curve(data[i][:, label_column], data[i][:, score_column])
        recall_array.append(recall)
        precision_array.append(precision)
        precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1])[::-1])
        roc_auc = auc(recall, precision)
        aucs.append(roc_auc)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'blueviolet', 'deeppink', 'cyan'])
    # ROC plot for CV
    fig = plt.figure(figsize=(7,7))
    for i, color in zip(range(len(recall_array)), colors):
        if datasize>1:
            plt.plot(recall_array[i], precision_array[i], lw=1, alpha=0.7, color=color,
                     label='PRC fold %d (AUPRC = %0.4f)' % (i + 1, aucs[i]))
        else:
            plt.plot(recall_array[i], precision_array[i], lw=1, alpha=0.7, color=color,
                     label='PRC (AUPRC = %0.4f)' %aucs[i])
    mean_precision = np.mean(precisions, axis=0)
    mean_recall = mean_recall[::-1]
    mean_auc = auc(mean_recall, mean_precision)
    std_auc = np.std(aucs)
    
    if datasize>1:
        plt.plot(mean_recall, mean_precision, color='blue',
                 label=r'Mean PRC (AUPRC = %0.4f $\pm$ %0.3f)' % (mean_auc, std_auc),
                 lw=2, alpha=.9)
    std_precision = np.std(precisions, axis=0)
    precision_upper = np.minimum(mean_precision + std_precision, 1)
    precision_lower = np.maximum(mean_precision - std_precision, 0)
    if datasize>1:
        plt.fill_between(mean_recall, precision_lower, precision_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.savefig(output)
    plt.close(0)
    return mean_auc, aucs
    
# Calculate and save performance metrics
def calculate_metrics(labels, scores, cutoff=0.5, po_label=1):
    my_metrics = {
        'SN': 'NA',
        'SP': 'NA',
        'ACC': 'NA',
        'MCC': 'NA',
        'Recall': 'NA',
        'Precision': 'NA',
        'F1-score': 'NA',
        'Cutoff': cutoff,
    }

    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(scores)):
        if labels[i] == po_label:
            if scores[i] >= cutoff:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if scores[i] < cutoff:
                tn = tn + 1
            else:
                fp = fp + 1

    my_metrics['SN'] = tp / (tp + fn) if (tp + fn) != 0 else 'NA'
    my_metrics['SP'] = tn / (fp + tn) if (fp + tn) != 0 else 'NA'
    my_metrics['ACC'] = (tp + tn) / (tp + fn + tn + fp)
    my_metrics['MCC'] = (tp * tn - fp * fn) / np.math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (
                                                                                                                     tp + fp) * (
                                                                                                                     tp + fn) * (
                                                                                                                     tn + fp) * (
                                                                                                                     tn + fn) != 0 else 'NA'
    my_metrics['Precision'] = tp / (tp + fp) if (tp + fp) != 0 else 'NA'
    my_metrics['Recall'] = my_metrics['SN']
    my_metrics['F1-score'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 'NA'
    return my_metrics

def calculate_metrics_list(data, label_column=0, score_column=2, cutoff=0.5, po_label=1):
    metrics_list = []
    for i in data:
        metrics_list.append(calculate_metrics(i[:, label_column], i[:, score_column], cutoff=cutoff, po_label=po_label))
    if len(metrics_list) == 1:
        return metrics_list
    else:
        mean_dict = {}
        std_dict = {}
        keys = metrics_list[0].keys()
        for i in keys:
            mean_list = []
            for metric in metrics_list:
                mean_list.append(metric[i])
            mean_dict[i] = np.array(mean_list).sum() / len(metrics_list)
            std_dict[i] = np.array(mean_list).std()
        metrics_list.append(mean_dict)
        metrics_list.append(std_dict)
        return metrics_list
        
def save_prediction_metrics_list(metrics_list, output):
    if len(metrics_list) == 1:
        with open(output, 'w') as f:
            f.write('Result')
            for keys in metrics_list[0]:
                f.write('\t%s' % keys)
            f.write('\n')
            for i in range(len(metrics_list)):
                f.write('value')
                for keys in metrics_list[i]:
                    f.write('\t%s' % metrics_list[i][keys])
                f.write('\n')
            f.close()
    else:
        with open(output, 'w') as f:
            f.write('Fold')
            for keys in metrics_list[0]:
                f.write('\t%s' % keys)
            f.write('\n')
            for i in range(len(metrics_list)):
                if i <= len(metrics_list)-3:
                    f.write('%d' % (i + 1))
                elif i == len(metrics_list)-2:
                    f.write('mean')
                else:
                    f.write('std')
                for keys in metrics_list[i]:
                    f.write('\t%s' % metrics_list[i][keys])
                f.write('\n')
            f.close()
    return None
    
# Fixed SP value, computing performance
def fixed_sp_calculate_metrics_list(data, cutoffs, label_column=0, score_column=1, po_label=1):
    metrics_list = []
    for index,i in enumerate(data):
        metrics_list.append(calculate_metrics(i[:, label_column], i[:, score_column], cutoff=cutoffs[index], po_label=po_label))
    if len(metrics_list) == 1:
        return metrics_list
    else:
        mean_dict = {}
        std_dict = {}
        keys = metrics_list[0].keys()
        for i in keys:
            mean_list = []
            for metric in metrics_list:
                mean_list.append(metric[i])
            mean_dict[i] = np.array(mean_list).sum() / len(metrics_list)
            std_dict[i] = np.array(mean_list).std()
        metrics_list.append(mean_dict)
        metrics_list.append(std_dict)
        return metrics_list