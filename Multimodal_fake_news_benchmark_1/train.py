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
from eval import eval



def train(train_iter, dev_dataloader, model, args):
    if args.cuda:
        print("training model in cuda ...")
        model.cuda()

    print(f'training using {args.model_name}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_epoch = 0
    print(model)
    cur_save_path = None
    for epoch in range(1, args.epochs+1):
        model.train()
        losses = []
        for batch in train_iter:
            train_input, target = batch[0], batch[1]
            mask = train_input['attention_mask']
            input_id = train_input['input_ids'].squeeze(1)
            #feature.data.t_(), target.data.sub_(1)  # batch first, index align
            if args.cuda:
                target = target.cuda()
                mask = mask.cuda()
                input_id = input_id.cuda()

            optimizer.zero_grad()
            logit = model(input_id, mask)
            # print(logit.shape, target.shape)
            loss = F.cross_entropy(logit, target)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.data)

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum() # training_batch statistics
                accuracy = 100.0 * corrects / args.batch_size
                try:
                    sys.stdout.write(
                        '\rEpoch[{}] Batch[{}] - loss: {:.4f} '
                        'acc: {:.4f}%({}/{})'.format(epoch,
                                                     steps,
                                                     loss.item(),
                                                     accuracy,
                                                     corrects,
                                                     args.batch_size))
                except:
                    print("Unexpected error:", sys.exc_info()[0])

        if epoch % args.test_interval == 0:
            dev_acc, dev_loss = eval(dev_dataloader, model, args) # need check eval

            print('Train loss:', np.mean(losses), 'Val loss:', dev_loss)
            if dev_acc > best_acc:
                best_acc = dev_acc
                last_epoch = epoch
                if args.save_best:
                    if cur_save_path is not None:
                        os.remove(cur_save_path)
                    cur_save_path = save(model, args.save_dir, 'best_'+args.bert_type, steps)
                print(f"the current best dev results appear at epoch {epoch}")
            else:
                if epoch - last_epoch >= args.early_stop:
                    print('early stop by {} steps.'.format(args.early_stop))
                    break
        # if steps % args.save_interval == 0:
        #     save(model, args.save_dir, 'snapshot', steps)


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix1 = os.path.join(save_dir, save_prefix)
    tmp = save_prefix.split('/')
    cur_dir = save_dir
    for t in tmp[:-1]:
        cur_dir = os.path.join(cur_dir, t)
        if not os.path.isdir(cur_dir):
            os.makedirs(cur_dir)
    save_path = '{}_steps_{}.pt'.format(save_prefix1, steps)
    torch.save(model.state_dict(), save_path)
    return save_path