import os
import sys
from operator import itemgetter

import sklearn
import sklearn.metrics
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

## general evaluation
def eval(val_iter, model, args):
    print("using normal evaluation ...")
    model.eval()
    corrects, avg_loss = 0, 0
    #print(model)
    y_pred = []
    y_truth = []
    logit_diff = []
    logit_var = []
    losses = []
    represent_all = torch.FloatTensor()
    target_all = torch.LongTensor()

    for batch in val_iter:
        val_input, target = batch[0], batch[1]
        if args.cuda:
            target = target.cuda()
            val_input['input_ids'] = val_input['input_ids'].cuda()
            val_input['attention_mask'] = val_input['attention_mask'].cuda()
            val_input['pixel_values'] = val_input['pixel_values'].cuda()
            if args.clip_type.find('visualbert') != -1:
                val_input['token_type_ids'] = val_input['token_type_ids'].cuda()
                logit = model(val_input['input_ids'], val_input['attention_mask'], val_input['pixel_values'], val_input['token_type_ids'])
            else:
                logit = model(val_input['input_ids'], val_input['attention_mask'], val_input['pixel_values'])
        loss = F.cross_entropy(logit, target)
        losses.append(loss.item())


        y_pred_cur = (torch.max(logit, 1)[1].view(target.size()).data).tolist()
        y_truth_cur = target.data.tolist()

        y_pred += y_pred_cur
        y_truth += y_truth_cur

        logit_var_cur = np.var(logit.cpu().data.numpy(), axis=1).tolist()
        logit_var += logit_var_cur
        logit_diff_cur = (logit[:, 1] - logit[:, 0]).data.tolist()
        logit_diff += logit_diff_cur

        target_all = torch.cat([target_all, target.data.cpu()], 0)



    # print(logit_diff)
    f1_score = show_results(y_truth, y_pred)

    return f1_score, np.mean(losses)

 
def show_results(y_truth, y_pred):

    # print(np.sum(y_truth), np.sum(y_pred))
    accuracy_score = (sklearn.metrics.accuracy_score(y_truth, y_pred))
    f1_score = (sklearn.metrics.f1_score(y_truth, y_pred, pos_label=1))
    f1_ma = (sklearn.metrics.f1_score(y_truth, y_pred, pos_label=1, average='macro'))
    f1_mi = (sklearn.metrics.f1_score(y_truth, y_pred, pos_label=1, average='micro'))
    prec_score = (sklearn.metrics.precision_score(y_truth, y_pred, pos_label=1))
    recall_score = (sklearn.metrics.recall_score(y_truth, y_pred, pos_label=1))
    # confusion_mat = (sklearn.metrics.confusion_matrix(y_truth, y_pred))

    print(f"accuracy_score={accuracy_score}, f1_score={f1_score}, prec_score={prec_score}, recall_score={recall_score}, f1_ma={f1_ma}, f1_mi={f1_mi}")
    #print(confusion_mat)
    return f1_score

    # elif class_num > 2:
    #     accuracy_score = (sklearn.metrics.accuracy_score(y_truth, y_pred))
    #     micro_f1_score = (sklearn.metrics.f1_score(y_truth, y_pred, average='micro'))
    #     macro_f1_score = (sklearn.metrics.f1_score(y_truth, y_pred, average='macro'))
    #     print(accuracy_score, micro_f1_score, macro_f1_score, sep='\t')
    #     return micro_f1_score

