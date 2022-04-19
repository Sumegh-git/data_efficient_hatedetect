import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
import sys
import random
import math
import time
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from transformers import BertTokenizer, AutoTokenizer
from transformers import BertModel, AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup

from torch.utils.tensorboard import SummaryWriter

use_cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# use_cuda=False
# device='cpu'
np.random.seed(0)
torch.manual_seed(0)

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True

base_model = 'twitter-roberta-base'
model_list = ['bert-base-uncased', 'bert-base-multilingual-uncased', 'google/muril-base-cased', 'xlm-roberta-base',
              'ai4bharat/indic-bert','cardiffnlp/twitter-xlm-roberta-base','cardiffnlp/twitter-xlm-roberta-base-sentiment',
              'cardiffnlp/twitter-roberta-base', 'cardiffnlp/twitter-roberta-base-sentiment',
              'cardiffnlp/twitter-roberta-base-hate', 'roberta-base']

lang = 'hx'
# model_choice = 7
layer = int(sys.argv[1])

class HateData(Dataset):
    def __init__(self, data_path, split='train', lang='bengali', layer=1):
          
        self.layer = layer
        self.data = np.load("/mnt/" + lang + "_states_" + split + "_Tw.npy", allow_pickle=True)
        self.df = pd.read_csv(data_path + lang + "_" + split + ".tsv", sep='\t')

    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        emb = self.data[index][self.layer]
        labels = self.df.iloc[index][1]
        
        emb = torch.tensor(emb, dtype=torch.float).view(768)
        labels = torch.tensor(labels, dtype=torch.long).view(1)


        return emb, labels

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        H1, H2, num_class = 768, 128, 3

        self.clf = nn.Sequential(nn.Linear(H1, H2), nn.ReLU(), nn.Linear(H2, num_class))

        
    def forward(self, emb):  
        logits = self.clf(emb) # (batch, num_class)
        return logits

loss_fn = nn.CrossEntropyLoss()

def train(emb, label, model, model_opt, scdl):

    model_opt.zero_grad()

    batch_size = emb.shape[0]

    loss = 0.0

    if use_cuda:
        emb = emb.to(device)
        label = label.to(device)

    label = label.flatten()
    
    logits = model(emb)

    loss = loss_fn(logits, label)

    # if torch.isnan(loss):
    #     pass
    # else:
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip gradients to prevent exploding
    model_opt.step()
    scdl.step()
    # print(loss)
    return float(loss.item())

def evaluate(emb, label, model, mode='train'):
   
    batch_size = emb.shape[0]

    with torch.no_grad():
        if use_cuda:
            emb = emb.to(device)
            label = label.to(device)

        label = label.flatten()
        
        logits = model(emb)
        # print(nn.Softmax(dim=1)(logits))
        # print(label)
        loss = loss_fn(logits, label)
        
#         if mode == 'train':
#             return float(loss.item())
        
        preds = torch.argmax(logits, dim=1).flatten()
        # acc = (preds == label).cpu().numpy().mean() * 100

        return float(loss.item()), preds.cpu().numpy()
        

# df_test = pd.read_csv("/home/jupyter/data/implicit-hate-corpus/latent_test.tsv", sep='\t')
# gt_labels = np.array(df_test['class'])

# df_train = pd.read_csv("/home/jupyter/data/implicit-hate-corpus/latent_train.tsv", sep='\t')
# train_labels = np.array(df_train['class'])
df_test = pd.read_csv("/home/jupyter/data/test_data/hx_test.tsv", sep='\t')
gt_labels = np.array(df_test['label'])

# df_train = pd.read_csv("/home/jupyter/data/train_data/hx_train.tsv", sep='\t')
# train_labels = np.array(df_train['label'])

def trainIters(model, epochs, train_loader, test_loader, learning_rate=1e-3, log_step=240, valid_step=240, mode='train'):

    model_opt = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    num_train_steps = (len(train_loader)*epochs) 
    scdl = get_linear_schedule_with_warmup(model_opt, num_warmup_steps=int(0.1*num_train_steps), num_training_steps=num_train_steps)

    print("Initialised optimizer and lr scheduler")

    # valid_best_loss = [] 
    best_acc = 0.0
    test_ep = 0
    best_train_acc = 0.0

    tot = len(train_data) // train_loader.batch_size
    tot_val = len(val_data) // test_loader.batch_size
    plot_steps = 0
    
    for epoch in range(epochs):
        train_loss_total = 0.0
        train_step = 0
        # Training
        
        model.train()  
        for entry in (train_loader):
            loss = train(entry[0], entry[1], model, model_opt, scdl)
            plot_steps += 1
            train_step += 1
            # if not math.isnan(loss) :      
            train_loss_total = train_loss_total + loss
            
            train_loss = train_loss_total / train_step
            
            
         
        
        model.eval()
#         train_pred = []
#         for entry in (train_loader):
#             loss_v, pred_v = evaluate(entry[0], entry[1], model, mode='test')
#             # if not math.isnan(loss) :      
#             train_pred.extend([pd for pd in pred_v])
 
#         train_acc = f1_score(train_labels, train_pred, average='macro') 
        
        test_pred = []

        for entry in (test_loader):
            loss_v, pred_v = evaluate(entry[0], entry[1], model, mode='test')
            # if not math.isnan(loss) :      
            test_pred.extend([pd for pd in pred_v])

        # val_acc = (test_pred == gt_labels).mean().item()
        val_acc = f1_score(gt_labels, test_pred, average='macro')


        if val_acc > best_acc:
            best_acc = val_acc
            test_ep = epoch
            
        # best_train_acc = max(best_train_acc, train_acc)

    print("Best test F1: ", (best_acc, test_ep))
    # print("Best train F1: ", best_train_acc)
    
# train_data = HateData(data_path="/home/jupyter/data/implicit-hate-corpus/", split='train', lang=lang, layer=layer)
# val_data = HateData(data_path="/home/jupyter/data/implicit-hate-corpus/", split='test', lang=lang, layer=layer)
train_data = HateData(data_path="/home/jupyter/data/train_data/", split='train', lang=lang, layer=layer)
val_data = HateData(data_path="/home/jupyter/data/test_data/", split='test', lang=lang, layer=layer)

BS = 64
dataload = DataLoader(train_data, batch_size=BS, shuffle=True)
dataload_val = DataLoader(val_data, batch_size=BS, shuffle=False)

model = Classifier()
model = model.float()
# model = nn.DataParallel(model)#, device_ids = [2, 3]
model = model.to(device)

trainIters(model, 100, dataload, dataload_val)
        


## Code to pre-save layer-wise bert features for faster dataloading

# # data = pd.read_csv("/home/jupyter/data/train_data/hx_train.tsv", sep='\t')
# data = pd.read_csv("/home/jupyter/data/test_data/hx_test.tsv", sep='\t')
# # data = pd.read_csv("/home/jupyter/data/implicit-hate-corpus/latent_train.tsv", sep='\t')
# # data = pd.read_csv("/home/jupyter/data/implicit-hate-corpus/latent_test.tsv", sep='\t')

# MAX_SEQ_LEN = 128
# label_idx = 1
# text_idx = 0


# tokenizer = AutoTokenizer.from_pretrained(model_list[model_choice])
# bert = AutoModel.from_pretrained(model_list[model_choice])
# bert = bert.to(device)
# bert.eval()

# h_states = torch.zeros((len(data), 13, 768)).to(device)

# for index in tqdm(range(len(data)), position=0, leave=True):
#     row = data.iloc[index]

#     labels = row[label_idx]
#     text = str(row[text_idx])

#     inputs = tokenizer(text, padding='max_length', truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt").to(device)
#     # print(inputs)
#     with torch.no_grad():
#         outputs = bert(**inputs, output_hidden_states=True)
        
#     for layer in range(13):
#         emb = outputs[2][layer].mean(1).squeeze()
#         h_states[index][layer] = emb
        
# np.save("/mnt/hx_states_test_Tws.npy", h_states.cpu().numpy())