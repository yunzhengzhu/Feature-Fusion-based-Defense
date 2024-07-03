# import dependencies
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import csv
from torchmetrics import AUROC
import torch
import torch.nn as nn

def plot_tv(
        train_metric,
        val_metric,
        path,
        model_title,
        filename,
        metric
):
    """
    Description: This function plots training and validation curves for your DL model
    Param:
        fit_model = variable storing fit DL model from Keras
        path = folder you would like to save pictures
        model_title = keyword (string) to define model
        metrics = metrics list from keras model
    Return:
        Training and validation curves (loss and accuracy) for your DL model
        Figure will be saved with title [Timestamp_keyword]
    """
    # Plot the results
    if metric == "Accuracy":
        train_acc = train_metric
        val_acc = val_metric
        fig1,(fig_acc) = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
        xa = np.arange(len(val_acc))
        fig_acc.plot(xa, train_acc)
        fig_acc.plot(xa, val_acc)
        fig_acc.set(xlabel='Epochs')
        fig_acc.set(ylabel='Accuracy')
        fig_acc.set(title='Training Accuracy vs Validation Accuracy')
        fig1.legend(['Train', 'Validation'], loc=1, borderaxespad=1)
        #fig.grid('True')
        fig1.savefig((os.path.join(path, (filename + '_' + metric + '_' + model_title + '.png'))))
    elif metric =="AUC":
        train_acc = train_metric
        val_acc = val_metric
        fig2,(fig_acc) = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
        xa = np.arange(len(val_acc))
        fig_acc.plot(xa, train_acc)
        fig_acc.plot(xa, val_acc)
        fig_acc.set(xlabel='Epochs')
        fig_acc.set(ylabel='AUC')
        fig_acc.set(title='Training AUC vs Validation AUC')
        fig2.legend(['Train', 'Validation'], loc=1, borderaxespad=.5)
        #fig.grid('True')
        fig2.savefig((os.path.join(path, (filename + '_' + metric + '_' + model_title + '.png'))))
    elif metric == "Loss":
        train_loss = train_metric
        val_loss = val_metric
        fig3, (fig_loss) = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
        xc = np.arange(len(train_loss))
        fig_loss.plot(xc, train_loss)
        fig_loss.plot(xc, val_loss)
        fig_loss.set(xlabel='Epochs')
        fig_loss.set(ylabel='Loss')
        fig_loss.set(title='Training Loss vs Validation Loss')
        fig3.legend(['Train', 'Validation'], loc=1, borderaxespad=.5)
        #fig.grid('True')
        fig3.savefig((os.path.join(path, (filename + '_loss_' + model_title + '.png'))))
    else: print("ERROR: NOT GRAPHED-", i)


def plot_attacks_all(classifier, x, y, path_fig, path_csv, dataset, title, num_classes):
    from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent
    eps_range = [0.0001, 0.0003, 0.0006, 0.0008, 0.001, 0.00125, 0.0015, 0.00175, 0.002, 0.0023, 0.0026, 0.0028, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02]
    step_size = 0.001
    nb_correct_fgsm, nb_roc_fgsm = [], []
    nb_correct_pgd, nb_roc_pgd = [], []
    nb_correct_bim, nb_roc_bim = [], []
    for eps in eps_range:
        attacker_fgsm = FastGradientMethod(classifier, eps=eps)
        attacker_pgd = ProjectedGradientDescent(classifier, eps=eps, eps_step=eps/4, max_iter=10, num_random_init=5)
        attacker_bim = BasicIterativeMethod(classifier, eps=eps, eps_step=eps/10, max_iter=10)
        x_fgsm = attacker_fgsm.generate(x)
        x_pgd = attacker_pgd.generate(x)
        x_bim = attacker_bim.generate(x)
        
        pred_fgsm = classifier.predict(x_fgsm)
        x_pred_fgsm = np.argmax(pred_fgsm, axis=1)
        nb_correct_fgsm += [np.sum(x_pred_fgsm == np.argmax(y, axis=1))]
        nb_roc_fgsm += [
            AUROC(task="multiclass", num_classes=num_classes)(
                nn.Softmax(dim=1)(torch.from_numpy(pred_fgsm)),
                torch.from_numpy(np.where(y)[1]).long(),
            )
        ]
        
        pred_pgd = classifier.predict(x_pgd)
        x_pred_pgd = np.argmax(pred_pgd, axis=1)
        nb_correct_pgd += [np.sum(x_pred_pgd == np.argmax(y, axis=1))]
        nb_roc_pgd += [
            AUROC(task="multiclass", num_classes=num_classes)(
                nn.Softmax(dim=1)(torch.from_numpy(pred_pgd)),
                torch.from_numpy(np.where(y)[1]).long(),
            )
        ]

        pred_bim = classifier.predict(x_bim)
        x_pred_bim = np.argmax(pred_bim, axis=1)
        nb_correct_bim += [np.sum(x_pred_bim == np.argmax(y, axis=1))]
        nb_roc_bim += [
            AUROC(task="multiclass", num_classes=num_classes)(
                nn.Softmax(dim=1)(torch.from_numpy(pred_bim)),
                torch.from_numpy(np.where(y)[1]).long(),
            )
        ]


    fig, ax = plt.subplots()
    ax.plot(np.array(eps_range) / step_size, np.array(nb_correct_fgsm) / y.shape[0], 'b--', label='FGSM')
    ax.plot(np.array(eps_range) / step_size, np.array(nb_correct_pgd) / y.shape[0], 'r--', label='PGD')
    ax.plot(np.array(eps_range) / step_size, np.array(nb_correct_bim) / y.shape[0], 'g--', label='BIM')
    legend = ax.legend(loc='upper right', shadow=True, fontsize='large')
    legend.get_frame().set_facecolor('#FFFFFF')
    plt.xlabel('Perturbation (x $10^{-3}$)')
    plt.ylabel('Accuracy (%)')
    plt.savefig(path_fig + dataset + '/' + title + 'acc.png')
    plt.clf()
    
    fig, ax = plt.subplots()
    ax.plot(np.array(eps_range) / step_size, np.array(nb_roc_fgsm), 'b--', label='FGSM')
    ax.plot(np.array(eps_range) / step_size, np.array(nb_roc_pgd), 'r--', label='PGD')
    ax.plot(np.array(eps_range) / step_size, np.array(nb_roc_bim), 'g--', label='BIM')
    legend = ax.legend(loc='upper right', shadow=True, fontsize='large')
    legend.get_frame().set_facecolor('#FFFFFF')
    plt.xlabel('Perturbation (x $10^{-3}$)')
    plt.ylabel('AUROC')
    plt.savefig(path_fig + dataset + '/' + title + 'roc.png')
    plt.clf()

    data = [np.array(eps_range), np.array(nb_correct_fgsm) / y.shape[0], np.array(nb_correct_pgd) / y.shape[0], np.array(nb_correct_bim) / y.shape[0], np.array(nb_roc_fgsm), np.array(nb_roc_pgd), np.array(nb_roc_bim)]
    out = csv.writer(open(path_csv + dataset + '/' + title + '.csv', "w"), delimiter=',', quoting=csv.QUOTE_ALL)
    out.writerows(zip(*data))

def plot_roc_curve(fpr, tpr, auroc, save_path, part):
    '''
    part: train/val/test
    '''
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='orange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve ({:.3f})'.format(auroc))
    plt.legend(loc="lower right")
    
    plt.savefig(os.path.join(save_path, f'roc_curve_{part}.png'))
    plt.show()

