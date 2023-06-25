from torch import nn
import torch
# from transformers import BertModel, AutoTokenizer, AutoModel, pipeline
from transformers import CLIPModel, AutoModel, VisualBertModel, LxmertModel, ResNetModel



class CLIPClassifier(nn.Module):
    """
    refer to the usage here: https://huggingface.co/docs/transformers/model_doc/clip
    """
    def __init__(self, args):

        super(CLIPClassifier, self).__init__()
        if args.clip_type.find('clip') != -1:
            self.model = CLIPModel.from_pretrained(args.clip_type)
        elif args.clip_type.find('visualbert') != -1:
            self.visual_model = ResNetModel.from_pretrained('Ramos-Ramos/vicreg-resnet-50')
            self.model = VisualBertModel.from_pretrained(args.clip_type)
        elif args.clip_type.find('lxmert') != -1:
            self.model = LxmertModel.from_pretrained(args.clip_type)
        else:
            self.model = AutoModel.from_pretrained(args.clip_type)
        self.type = args.clip_type
        self.dropout = nn.Dropout(args.dropout)
        if args.clip_type.find('visualbert') != -1:
            self.l1 = nn.Linear(768, 256)
        else:
            self.l1 = nn.Linear(1024, 256)
        self.l2 = nn.Linear(256, 2)
        self.relu = nn.ReLU()

        if args.freeze_bert == True:
            for parameter in self.model.parameters():
                parameter.require_gard = False
            if args.clip_type.find('visualbert') != -1:
                for parameter in self.visual_model.parameters():
                    parameter.require_gard = False

    def forward(self, input_id, mask, pixel_value, token_id=None):

        # image_features = self.clip_model.encode_image(pixel_value)
        # text_features = self.clip_model.encode_text(input_id)
        # pooled_output = torch.cat([image_features, text_features])
        input_id, mask = input_id.squeeze(), mask.squeeze()

        if self.type.find('clip') != -1:
            output = self.model(input_ids=input_id, attention_mask=mask, pixel_values=pixel_value)
            pooled_output = torch.cat([output.image_embeds, output.text_embeds], dim=-1)
        else:
            token_id, mask = token_id.squeeze(), mask.squeeze()
            visual_embeds = self.visual_model(pixel_values=pixel_value).last_hidden_state
            visual_embeds = visual_embeds.permute(0, 2, 3, 1).reshape(visual_embeds.shape[0], -1, visual_embeds.shape[1]) # (batch_size, visual_seq_length, visual_embedding_dim)
            output = self.model(input_ids=input_id, token_type_ids=token_id, attention_mask=mask, visual_embeds=visual_embeds).last_hidden_state
            pooled_output = output.mean(dim=1).squeeze()


        dropout_output = self.dropout(pooled_output)
        dropout_output = self.relu(self.l1(dropout_output))
        linear_output = self.l2(dropout_output)

        return linear_output