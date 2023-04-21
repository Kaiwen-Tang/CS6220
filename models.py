from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from torch import nn
class Gen0Model(nn.Module):
    def __init__(self,output_dim,dropout_rate,freeze=False):
        super(Gen0Model,self).__init__()
        self.encoder=AutoModelForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states=True, return_dict=True)
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.dropout=nn.Dropout(dropout_rate)
        self.classifier=nn.Linear(3072,output_dim)
    
    def forward(self,input_ids,token_type_ids,attention_mask):
        outputs = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        hidden_states = torch.cat(tuple([outputs.hidden_states[i] for i in [-1, -2, -3, -4]]), dim=-1) # [bs, seq_len, hidden_dim*4]
        first_hidden_states = hidden_states[:, 0, :]
        x=self.dropout(first_hidden_states)
        x=self.classifier(x)
        return x

class Gen1Model(nn.Module):
    def __init__(self,output_dim,dropout_rate,freeze=False):
        super(Gen1Model,self).__init__()
        self.encoder=AutoModelForMaskedLM.from_pretrained("roberta-base", output_hidden_states=True, return_dict=True)
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.dropout=nn.Dropout(dropout_rate)
        self.classifier=nn.Linear(3072,output_dim)
    
    def forward(self,input_ids,token_type_ids,attention_mask):
        outputs = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # outputs = self.encoder(input_ids, attention_mask=attention_mask)
        hidden_states = torch.cat(tuple([outputs.hidden_states[i] for i in [-1, -2, -3, -4]]), dim=-1) # [bs, seq_len, hidden_dim*4]
        first_hidden_states = hidden_states[:, 0, :]
        x=self.dropout(first_hidden_states)
        x=self.classifier(x)
        return x

class Gen2Model(nn.Module):
    def __init__(self,output_dim,dropout_rate,freeze=False):
        super(Gen2Model,self).__init__()
        self.encoder=AutoModelForMaskedLM.from_pretrained("distilroberta-base", output_hidden_states=True, return_dict=True)
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.dropout=nn.Dropout(dropout_rate)
        self.classifier=nn.Linear(3072,output_dim)
    
    def forward(self,input_ids,token_type_ids,attention_mask):
        outputs = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        hidden_states = torch.cat(tuple([outputs.hidden_states[i] for i in [-1, -2, -3, -4]]), dim=-1) # [bs, seq_len, hidden_dim*4]
        first_hidden_states = hidden_states[:, 0, :]
        x=self.dropout(first_hidden_states)
        x=self.classifier(x)
        return x

class FinModel(nn.Module):
    def __init__(self,output_dim,dropout_rate,freeze=False):
        super(FinModel,self).__init__()
        self.encoder=AutoModelForMaskedLM.from_pretrained("ahmedrachid/FinancialBERT", output_hidden_states=True, return_dict=True)
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.dropout=nn.Dropout(dropout_rate)
        self.classifier=nn.Linear(3072,output_dim)
    
    def forward(self,input_ids,token_type_ids,attention_mask):
        outputs = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        hidden_states = torch.cat(tuple([outputs.hidden_states[i] for i in [-1, -2, -3, -4]]), dim=-1) # [bs, seq_len, hidden_dim*4]
        first_hidden_states = hidden_states[:, 0, :]
        x=self.dropout(first_hidden_states)
        x=self.classifier(x)
        return x

class CliModel(nn.Module):
    def __init__(self,output_dim,dropout_rate,freeze=False):
        super(CliModel,self).__init__()
        self.encoder=AutoModelForMaskedLM.from_pretrained("climatebert/distilroberta-base-climate-f", output_hidden_states=True, return_dict=True)
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.dropout=nn.Dropout(dropout_rate)
        self.classifier=nn.Linear(3072,output_dim)
    
    def forward(self,input_ids,token_type_ids,attention_mask):
        outputs = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        hidden_states = torch.cat(tuple([outputs.hidden_states[i] for i in [-1, -2, -3, -4]]), dim=-1) # [bs, seq_len, hidden_dim*4]
        first_hidden_states = hidden_states[:, 0, :]
        x=self.dropout(first_hidden_states)
        x=self.classifier(x)
        return x

  
