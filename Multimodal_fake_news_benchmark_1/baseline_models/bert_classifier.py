from torch import nn
from transformers import BertModel, AutoModel, pipeline, FunnelModel, AutoModelForMaskedLM, AlbertModel, RobertaModel, DistilBertModel

class BertClassifier(nn.Module):

    def __init__(self, args):

        super(BertClassifier, self).__init__()
        if args.bert_type.find('bert-base-cased') != -1:
            self.bert = BertModel.from_pretrained(args.bert_type)
        elif args.bert_type.find('BioBERT') != -1 or args.bert_type.find('declutr') != -1 \
                or args.bert_type.find('covid-twitter-bert-v2') != -1\
                or args.bert_type.find('all-MiniLM') != -1\
                or args.bert_type.find('distil') != -1:
            self.bert = AutoModel.from_pretrained(args.bert_type)
        elif args.bert_type.find('roberta') != -1:
            self.bert = RobertaModel.from_pretrained(args.bert_type)
        elif args.bert_type.find('funnel') != -1:
            self.bert = FunnelModel.from_pretrained(args.bert_type)
        elif args.bert_type.find('albert') != -1:
            self.bert = AlbertModel.from_pretrained(args.bert_type)
        elif args.bert_type.find('Fake_News') != -1:
            self.bert = DistilBertModel.from_pretrained(args.bert_type)
        self.type = args.bert_type
        self.dropout = nn.Dropout(args.dropout)
        if self.type.find('all-MiniLM') != -1:
            self.l1 = nn.Linear(384, 256)
        else:
            self.l1 = nn.Linear(768, 256)
        self.l2 = nn.Linear(256, 2)
        self.relu = nn.ReLU()

        if args.freeze_bert == True:
            for parameter in self.bert.parameters():
                parameter.require_gard = False

    def forward(self, input_id, mask):
        if self.type.find('funnel') != -1 or self.type.find('all-MiniLM') != -1:
            mask = mask.squeeze()
            pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)[0].mean(dim=1).squeeze()
        elif self.type.find('declutr') != -1:
            _, pooled_output, _ = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        elif self.type.find('Fake_News') != -1  or self.type.find('distil') != -1:
            pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)[0].mean(dim=1).squeeze()
        else:
            _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False) # pooled_output: text embeeding
        # print('pooled_output', pooled_output.shape)
        dropout_output = self.dropout(pooled_output)
        dropout_output = self.relu(self.l1(dropout_output))
        linear_output = self.l2(dropout_output)

        return linear_output