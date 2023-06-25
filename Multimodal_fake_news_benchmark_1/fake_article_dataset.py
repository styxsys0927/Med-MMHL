import nltk
import numpy as np
import itertools

import torch
from torch.autograd import Variable
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
import os.path
import glob
import ntpath
import json

from transformers import BertTokenizer, CLIPProcessor, CLIPImageProcessor, CLIPTokenizer, FunnelTokenizer, AutoTokenizer, \
    AlbertTokenizer, RobertaTokenizer, DistilBertTokenizer, LxmertTokenizer, AutoProcessor, AutoImageProcessor
# from sentence_transformers import SentenceTransformer
import clip
from PIL import Image

class PD_Dataset:

    def __init__(self, args, data_pandas):
        """Create an IMDB dataset instance. """

        #self.word_embed_file = self.data_folder + 'embedding/wiki.ar.vec'
        # word_embed_file = data_folder + "embedding/Wiki-CBOW"
        self.df_data = data_pandas
        self.class_num = 2

        if args.bert_type.find('funnel') != -1:
            self.tokenizer = FunnelTokenizer.from_pretrained(args.bert_type)
        elif args.bert_type.find('BioBERT') != -1 or args.bert_type.find('declutr') != -1 or args.bert_type.find('covid-twitter-bert-v2') != -1:
            self.tokenizer = AutoTokenizer.from_pretrained(args.bert_type)
        elif args.bert_type.find('roberta') != -1:
            self.tokenizer = RobertaTokenizer.from_pretrained(args.bert_type)
        elif args.bert_type.find('albert') != -1:
            self.tokenizer = AlbertTokenizer.from_pretrained(args.bert_type)
        elif args.bert_type.find('bert-base-cased') != -1:
            self.tokenizer = BertTokenizer.from_pretrained(args.bert_type)
        elif args.bert_type.find('Fake_News') != -1:
            self.tokenizer = DistilBertTokenizer.from_pretrained(args.bert_type)

        self.labels = [label for label in self.df_data['det_fake_label']]
        self.texts = []
        try:
            for text in self.df_data['content']:
                self.texts.append(self.tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt"))
        except Exception as e:
            print(e)
            print(text)

    # def __init__(self, df):
    #     self.labels = [labels[label] for label in df['category']]
    #     self.texts = [tokenizer(text,
    #                             padding='max_length', max_length=512, truncation=True,
    #                             return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

class PS_Dataset: # for per-sentence level classification

    def __init__(self, args, data_pandas):
        """Create an IMDB dataset instance. """

        #self.word_embed_file = self.data_folder + 'embedding/wiki.ar.vec'
        # word_embed_file = data_folder + "embedding/Wiki-CBOW"
        self.df_data = data_pandas
        self.class_num = 2

        if args.bert_type.find('all-MiniLM') != -1:
            self.tokenizer = AutoTokenizer.from_pretrained(args.bert_type)
        elif args.bert_type.find('distil') != -1:
            self.tokenizer = AutoTokenizer.from_pretrained(args.bert_type)

        self.labels = [label for label in self.df_data['det_fake_label']]
        self.texts = []
        try:
            for text in self.df_data['content']:
                self.texts.append(
                    self.tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt"))
        except:
            print(text)

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

class MM_Dataset: # multimodal dataset

    def __init__(self, args, data_pandas):
        self.df_data = data_pandas
        self.class_num = 2
        self.image_processor = None
        # self.processor = CLIPProcessor.from_pretrained(args.clip_type)
        if args.clip_type.find('lxmert') != -1:
            self.tokenizer =LxmertTokenizer.from_pretrained(args.clip_type)
        elif args.clip_type.find('clip') != -1:
            max_len = 77
            self.tokenizer =CLIPTokenizer.from_pretrained(args.clip_type)
        else:
            max_len = 49
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32')


        self.labels = [label for label in self.df_data['det_fake_label']]
        self.inputs = []

        try:
            ## load and process texts
            for i, row in self.df_data.iterrows():
                text, imgs = row['content'], row['image'][2:-2].split("', '")
                inputs = self.tokenizer(text, padding='max_length', max_length=max_len, truncation=True, return_tensors="pt")
                image = Image.fromarray(np.uint8(np.zeros((100, 100, 3))))
                for iid in range(len(imgs)):
                    try:
                        image = Image.open(imgs[iid])
                        break
                    except:
                        continue
                inputs['pixel_values'] = self.image_processor(images=image)['pixel_values'][0]
                self.inputs.append(inputs)

        except Exception as e:
            print(e, i)

        print('check dataset-----------------',len(self.labels), len(self.inputs))

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_inputs(self, idx):
        # Fetch a batch of inputs
        return self.inputs[idx]

    def __getitem__(self, idx):
        batch_inputs = self.get_batch_inputs(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_inputs, batch_y

