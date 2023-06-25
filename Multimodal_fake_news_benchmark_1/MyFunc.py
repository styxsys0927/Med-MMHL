
import pandas as pd
import os
import glob
import numpy as np
from PIL import Image
from sklearn.cluster import kmeans_plusplus
from sklearn.cluster import KMeans
import re

def read_benchmark_set(save_path):
    tr_sv_path = os.path.join(save_path, 'train.csv')
    dev_sv_path = os.path.join(save_path, 'dev.csv')
    te_sv_path = os.path.join(save_path, 'test.csv')

    tr_df = pd.read_csv(tr_sv_path, header=0, sep=',')
    dev_df = pd.read_csv(dev_sv_path, header=0, sep=',')
    te_df = pd.read_csv(te_sv_path, header=0, sep=',')

    # if 'source' in tr_df:
    #     tr_df = tr_df.sample(frac=0.2)  # shuffle operation
    #     dev_df = dev_df.sample(frac=0.2)  # shuffle operation
    #     te_df = te_df.sample(frac=0.2)  # shuffle operation
    print('Training set has {} fake news and {} true news'.format((tr_df['det_fake_label'] == 1).sum(),
                                                         (tr_df['det_fake_label'] == 0).sum()))
    print('Validation set has {} fake news and {} true news'.format((dev_df['det_fake_label'] == 1).sum(),
                                                         (dev_df['det_fake_label'] == 0).sum()))
    print('Testing set has {} fake news and {} true news'.format((te_df['det_fake_label'] == 1).sum(),
                                                         (te_df['det_fake_label'] == 0).sum()))
    return tr_df, dev_df, te_df
