# %debug
from black import main
import time
import random
import argparse
import pickle
import os
from tqdm import tqdm
from os.path import join
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.backends import cudnn

from fairseq.data import Dictionary
from fairseq.models.roberta import RobertaModel
from fairseq.data.encoders.fastbpe import fastBPE  
from fairseq import options  
from transformers.modeling_utils import * 
from transformers import *

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold



max_sequence_length = 128

data_path = 'Data/data_train.csv'

# Khởi tạo argument
EPOCHS = 20
BATCH_SIZE = 32
ACCUMULATION_STEPS = 5
FOLD = 4
LR = 2e-5
LR_DC_STEP = 80 
LR_DC = 0.1
NUM_WARMUP_STEPS = 10000
NUM_TRAIN_STEPS = 100000
CUR_DIR = os.path.dirname(os.getcwd())
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_PATH2 = 'model_ckpt2'



def _save_pkl(path, obj):
  with open(path, 'wb') as f:
    pickle.dump(obj, f)

def _load_pkl(path):
  with open(path, 'rb') as f:
    obj = pickle.load(f)
  return obj


def convert_lines(lines, vocab, bpe):
  '''
  lines: list các văn bản input
  vocab: từ điển dùng để encoding subwords
  bpe: 
  '''
  # Khởi tạo ma trận output
  outputs = np.zeros((len(lines), max_sequence_length), dtype=np.int32) # --> shape (number_lines, max_seq_len)
  # Index của các token cls (đầu câu), eos (cuối câu), padding (padding token)
  cls_id = 0
  eos_id = 2
  pad_id = 1

  for idx, row in tqdm(enumerate(lines), total=len(lines)): 
    # Mã hóa subwords theo byte pair encoding(bpe)
    subwords = bpe.encode('<s> '+ row +' </s>')
    input_ids = vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()
    # Truncate input nếu độ dài vượt quá max_seq_len
    if len(input_ids) > max_sequence_length: 
      input_ids = input_ids[:max_sequence_length] 
      input_ids[-1] = eos_id
    else:
      # Padding nếu độ dài câu chưa bằng max_seq_len
      input_ids = input_ids + [pad_id, ]*(max_sequence_length - len(input_ids))
    
    outputs[idx,:] = np.array(input_ids)
  return outputs

def evaluate(logits, targets):
    """
    Đánh giá model sử dụng accuracy và f1 scores.
    Args:
        logits (B,C): torch.LongTensor. giá trị predicted logit cho class output.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        acc (float): the accuracy score
        f1 (float): the f1 score
    """
    # Tính accuracy score và f1_score
    logits = logits.detach().cpu().numpy()    
    y_pred = np.argmax(logits, axis = 1)
    targets = targets.detach().cpu().numpy()
    f1 = f1_score(targets, y_pred, average='weighted')
    acc = accuracy_score(targets, y_pred)
    return acc, f1

def trainOnEpoch(train_loader, model, optimizer, epoch, num_epochs, criteria, device, log_aggr = 100):
    model.train()
    sum_epoch_loss = 0
    sum_acc = 0
    sum_f1 = 0
    start = time.time()
    for i, (x_batch, y_batch) in enumerate(train_loader):
      x_batch = x_batch.to(device)
      y_batch = y_batch.to(device)
      optimizer.zero_grad()
      y_pred = model.predict('new_task', x_batch)
      logits = torch.exp(y_pred)
      acc, f1 = evaluate(logits, y_batch)
      loss = criteria(y_pred, y_batch)
      loss.backward()
      optimizer.step()

      loss_val = loss.item()
      sum_epoch_loss += loss_val
      sum_acc += acc
      sum_f1 += f1
      iter_num = epoch * len(train_loader) + i + 1

      if i % log_aggr == 0:
            print('[TRAIN] epoch %d/%d  observation %d/%d batch loss: %.4f (avg %.4f),  avg acc: %.4f, avg f1: %.4f, (%.2f im/s)'
                % (epoch + 1, num_epochs, i, len(train_loader), loss_val, sum_epoch_loss / (i + 1),  sum_acc/(i+1), sum_f1/(i+1),
                  len(x_batch) / (time.time() - start)))
      start = time.time()  

def validate(valid_loader, model, device):
    model.eval()
    accs = []
    f1s = []
    with torch.no_grad():
        for x_batch, y_batch in valid_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model.predict('new_task', x_batch)
            logits = torch.exp(outputs)
            acc, f1 = evaluate(logits, y_batch)
            accs.append(acc)
            f1s.append(f1)
    
    mean_acc = np.mean(accs)
    mean_f1 = np.mean(f1s)
    return mean_acc, mean_f1




def main():

    # Load model pretrain `RoBERTa`
    from fairseq.models.roberta import RobertaModel
    phoBERT = RobertaModel.from_pretrained('PhoBERT_base_fairseq', checkpoint_file='model.pt')
    phoBERT.eval()  # disable dropout (or leave in train mode to finetune

    # Khởi tạo Byte Pair Encoding cho PhoBERT
    class BPE():
        bpe_codes = 'PhoBERT_base_fairseq/bpe.codes'
    args = BPE()
    phoBERT.bpe = fastBPE(args) #Incorporate the BPE encoder into PhoBERT
    phoBERT.to(DEVICE)

    # Load the dictionary  
    vocab = Dictionary()
    vocab.add_from_file("PhoBERT_base_fairseq/dict.txt")

    #Load data
    data = pd.read_csv(data_path)

    # tách dữ liệu ra làm đôi
    X = data.loc[:, ['pre_process_token_vncorenlp']]
    y = data.loc[:, ['sentiment']]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, random_state=0)       
    
    # Lưu lại các files
    _save_pkl('text_train.pkl', X_train)
    _save_pkl('label_train.pkl', y_train)
    _save_pkl('text_test.pkl', X_test)
    _save_pkl('label_test.pkl', y_test)

    
    # Tiếp theo ta sẽ tokenize các câu văn sang chuỗi index và padding câu văn về cũng một độ dài.

    X = convert_lines(X_train, vocab, phoBERT.bpe)
    lb = LabelEncoder()
    lb.fit(y_train)
    y = lb.fit_transform(y_train)
    # Save dữ liệu
    _save_pkl('PhoBERT_pretrain/X1.pkl', X)
    _save_pkl('PhoBERT_pretrain/y1.pkl', y)
    _save_pkl('PhoBERT_pretrain/labelEncoder1.pkl', lb)


    if not os.path.exists(CKPT_PATH2):
        os.mkdir(CKPT_PATH2)

    # Khởi tạo DataLoader
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(X, y))

    for fold, (train_idx, val_idx) in enumerate(splits):
        best_score = 0
        if fold != FOLD:
            continue
        print("Training for fold {}".format(fold))
        
        # Create dataset
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X[train_idx],dtype=torch.long), 
            torch.tensor(y[train_idx],dtype=torch.long))
        valid_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X[val_idx],dtype=torch.long), 
            torch.tensor(y[val_idx],dtype=torch.long))

        # Create DataLoader
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False)

        # Khởi tạo model:
        MODEL_LAST_CKPT = os.path.join(CKPT_PATH2, 'latest_checkpoint.pth.tar')
        if os.path.exists(MODEL_LAST_CKPT):
            print('Load checkpoint model!: ', MODEL_LAST_CKPT)
            phoBERT = torch.load(MODEL_LAST_CKPT)
        else:
            print('Load model pretrained!')
            phoBERT.register_classification_head('new_task', num_classes=3)
        
        ## Load BPE
        print('Load BPE')
        class BPE():
            bpe_codes = 'PhoBERT_base_fairseq/bpe.codes'
        args = BPE()
        phoBERT.bpe = fastBPE(args) #Incorporate the BPE encoder into PhoBERT
        phoBERT.to(DEVICE)


        # Khởi tạo optimizer và scheduler, criteria
        print('Init Optimizer, scheduler, criteria')
        param_optimizer = list(phoBERT.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.05},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.05}
        ]

        # num_train_optimization_steps = int(EPOCHS*len(train_dataset)/BATCH_SIZE/ACCUMULATION_STEPS)
        optimizer = AdamW(
            optimizer_grouped_parameters, 
            lr=LR, 
            correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=NUM_WARMUP_STEPS, 
            num_training_steps=NUM_TRAIN_STEPS)  # scheduler với linear warmup
        scheduler0 = get_constant_schedule(optimizer)  # scheduler với hằng số

        # criteria = nn.NLLLoss()
        criteria = nn.CrossEntropyLoss()

        avg_loss = 0.
        avg_accuracy = 0.
        frozen = True
        for epoch in tqdm(range(EPOCHS)):
            # warm up tại epoch đầu tiên, sau epoch đầu sẽ phá băng các layers
            if epoch > 0 and frozen:
                for child in phoBERT.children():
                    for param in child.parameters():
                        param.requires_grad = True
                frozen = False
                del scheduler0
                torch.cuda.empty_cache()
            # Train model on EPOCH
            print('Epoch: ', epoch)
            trainOnEpoch(
                train_loader=train_loader, 
                model=phoBERT, 
                optimizer=optimizer, 
                epoch=epoch, 
                num_epochs=EPOCHS, 
                criteria=criteria, 
                device=DEVICE, 
                log_aggr=100 )
            # scheduler.step(epoch = epoch)
            # Phá băng layers sau epoch đầu tiên
            if not frozen:
                scheduler.step()
            else:
                scheduler0.step()
            optimizer.zero_grad()
            # Validate on validation set
            acc, f1 = validate(valid_loader, phoBERT, device=DEVICE)
            print('Epoch {} validation: acc: {:.4f}, f1: {:.4f} \n'.format(epoch, acc, f1))

            # Store best model checkpoint
            ckpt_dict = {
                'epoch': epoch + 1,
                'state_dict': phoBERT.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            # Save model checkpoint into 'latest_checkpoint.pth.tar'
            torch.save(ckpt_dict, MODEL_LAST_CKPT)

if __name__ == "__main__":
    main()
