import os
import sys
import json 
import numpy as np

import torch
from sklearn.metrics import roc_auc_score


class Logger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_path = log_path
        f = open(self.log_path, "w")
        f.close()

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_path, "a") as f:
            f.write(message)

    def flush(self):
        pass


class AverageMeter(object):
    def __init__(self, limit=100):
        self.items = []
        self.limit = limit
        self.avg = 0

    def __repr__(self):
        return '{:.4f}'.format(self.avg)

    def toJSON(self):
        return json.dumps(self.avg)

    def update(self, val):
        self.items.append(val)
        if len(self.items) > self.limit:
            self.items = self.items[1:]
        self.avg = sum(self.items) / len(self.items)


def compute_accuracy(pred, label):
    if not isinstance(pred, np.ndarray):
        pred = pred.data.cpu().numpy()
    if not isinstance(label, np.ndarray):
        label = label.data.cpu().numpy()
    pred = pred.argmax(axis=1)
    correct = (pred == label).sum()
    acc = correct / len(label)
    return acc

def closed_set_accuracy(pred, label, known_classes):
    if not isinstance(pred, np.ndarray):
        pred = pred.data.cpu().numpy()
    if not isinstance(label, np.ndarray):
        label = label.data.cpu().numpy()
    pred = pred.argmax(axis=1)
    correct = (pred == label).sum()
    total = (label<known_classes).sum()
    acc = correct / total
    return acc

def compute_auroc(pred, label, known_classes):
    if not isinstance(pred, np.ndarray):
        pred = pred.data.cpu().numpy()
    if not isinstance(label, np.ndarray):
        label = label.data.cpu().numpy()

    scores = pred.max(axis=1)
    ood_labels = np.ones_like(label)
    ood_labels[label>=known_classes] = 0
    auroc = roc_auc_score(ood_labels, scores)
    return auroc





def save_result(result, predictions, save_dir):
    def _dumper(obj):
        try:
            return obj.toJSON()
        except:
            return obj.__dict__
    with open(os.path.join(save_dir, 'result.json'), 'w') as fp:
        json.dump(result, fp, default=_dumper, indent=4)

    with open(os.path.join(save_dir, 'predictions.txt'), "w") as f_out:
        for idx, preds in enumerate(predictions):
            f_out.write(f"{idx},{preds.argmax()},{preds.max()}\n")


def save_model(model, save_dir, postfix):
    model_path = os.path.join(save_dir, 'checkpoint_{}.pth'.format(postfix))
    model.cpu()
    print('save model to {}'.format(model_path))
    torch.save(model.state_dict(), model_path)
    model.cuda()
