import torch
from Config import *
from typing import Dict, List, Tuple
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from tqdm.auto import tqdm
import torchmetrics
import torchmetrics.functional as tmf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import csv
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from Config import *
import torchmetrics
# 随机过程的控制
np.random.seed(Config.random_seed)
torch.manual_seed(Config.random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.random_seed)


def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device):

  # Put model in train mode
  model.train()

  # Setup train loss and train accuracy values
  train_loss, train_acc = 0, 0

  # Loop through data loader data batches
  for batch, (X, y) in enumerate(dataloader):
      # Send data to target device
      X, y = X.to(device), y.to(device)

      # 1. Forward pass
      y_pred = model(X)

      # 2. Calculate  and accumulate loss
      loss = loss_fn(y_pred, y)
      train_loss += loss.item() 

      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backward
      loss.backward()

      # 5. Optimizer step
      optimizer.step()

      # Calculate and accumulate accuracy metric across all batches
      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  # Adjust metrics to get average loss and accuracy per batch 
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:

  # Put model in eval mode
  model.eval() 

  # Setup test loss and test accuracy values
  test_loss, test_acc = 0, 0

  # Turn on inference context manager
  with torch.inference_mode():
      # Loop through DataLoader batches
      for batch, (X, y) in enumerate(dataloader):
          # Send data to target device
          X, y = X.to(device), y.to(device)

          # 1. Forward pass
          test_pred_logits = model(X)

          # 2. Calculate and accumulate loss
          loss = loss_fn(test_pred_logits, y)
          test_loss += loss.item()

          # Calculate and accumulate accuracy
          test_pred_labels = test_pred_logits.argmax(dim=1)
          test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

  # Adjust metrics to get average loss and accuracy per batch 
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:

  results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
  }

  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
      train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
      test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

      # Print out what's happening
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
      )

      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)
  torch.save(model.state_dict(), 'F:/python/机器学习/model_dict/source.pth')
  # Return the filled results at the end of the epochs
  return results

def train_loop(model,train_dataloader,epochs,test_dataloader=None,evaluation=False):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                      lr=Config.lr,  # 默认学习率
                      eps=Config.eps  # 默认精度
                      )
    # 开始训练
    model.train()
    loss_epoch = []
    time_all = []
    acc_all = []
    for epoch in range(epochs):
        print(f"{'Epoch':^7}|{'40*Batch':^12}|{'train_Loss':^14}|{'train_accuracy':^14}|{'test_Loss':^14}|{'test_accuracy':^14}|{'时间':^9}")
        print("-" * 140)
        # 记录时间
        t0_epoch, t0_batch = time.time(), time.time()
        total_loss, batch_loss, batch_counts = 0, 0, 0
        train_accuracy = []
        for step,(X,y) in enumerate(train_dataloader):
            batch_counts +=1
            X = X.to(Config.device)
            y = y.to(Config.device)
            y_pre = model(X)
            loss = loss_fn(y_pre,y)
            batch_loss += loss.item()
            total_loss += loss.item()
            # 计算准确率
            y_pred_class = torch.argmax(torch.softmax(y_pre, dim=1), dim=1)
            train_accuracy.append((y_pred_class == y).sum().item()/len(y_pre))
            # 反向传播
            model.zero_grad()
            loss.backward()
            optimizer.step()
            # 每 40个batch输出一次
            if (step % 40 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # 计算40个batch的时间
                time_elapsed = time.time() - t0_batch
                train_accuracy = np.mean(train_accuracy)
                # Print训练结果
                print(f"{epoch+ 1:^7}|{step:^12}|{batch_loss / batch_counts:^14.6f}|{train_accuracy:^14.4f}%|{'-':^14}|{'-':^14}|{time_elapsed:^9.2f}")
                if len(time_all) == 0:
                    time_all.append(int(time_elapsed))
                else :
                    time_all.append(time_all[-1] + time_elapsed)
                loss_epoch.append(batch_loss / batch_counts)
                acc_all.append(train_accuracy)
                # 重置batch参数
                batch_loss, batch_counts = 0, 0
                train_accuracy = []
                t0_batch = time.time()

        # 计算平均loss 这个是训练集的loss
        avg_train_loss = total_loss / len(train_dataloader)
        
        print("-" * 80)
        if evaluation:  # 这个evalution是我们自己给的，用来判断是否需要我们汇总评估
            # 每个epoch之后评估一下性能
            # 在我们的验证集/测试集上.
            test_loss,test_accuracy= evaluate(model, test_dataloader)
            # Print 整个训练集的耗时
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch + 1:^7}|{'-':^12}|{avg_train_loss:^14.6f}|{'-':^14}|{test_loss:^14.6f}|{test_accuracy:^14.4f}%|{time_elapsed:^9.2f}")
            print("-" * 140)
    print(f"保存参数")
    save_path = Config.save_path
    torch.save(model.state_dict(), save_path)
    print(f"参数已保存\n地址:{save_path}")
    print("\n")     
    return loss_epoch , time_all,acc_all


def evaluate(model, test_dataloader):
    """
    在每个epoch后验证集上评估model性能
    """
    # model放入评估模式
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    # 准确率和误差
    test_accuracy = []
    test_loss = []

    # 验证集上的每个batch
    for (X,y) in test_dataloader:
        X = X.to(Config.device)
        y = y.to(Config.device)
        # 计算结果，不计算梯度
        with torch.no_grad():
            y_pre = model(X)  
        # 计算误差
        loss = loss_fn(y_pre, y)
        test_loss.append(loss.item())
        y_pred_class = torch.argmax(torch.softmax(y_pre, dim=1), dim=1)
        test_accuracy.append((y_pred_class == y).sum().item()/len(y_pre))
    val_loss = np.mean(test_loss)
    val_accuracy = np.mean(test_accuracy)

    return val_loss, val_accuracy


def loss_draw(loss,time,acc,title):
    x_draw = list(range(len(loss)))
    plt.figure(figsize=(10, 5))  
    plt.plot(time, loss, marker='o', linestyle='-', color='b', label='Loss') 
    plt.plot(time,acc, marker='*', linestyle='-', color='r', label='Accuracy')
    plt.title('Training Loss and Accuracy Over Time' + " " +title)
    plt.xlabel('Value')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    fig_name = "F:/python/机器学习/picture/" + title +'_'  +  '.png'
    plt.savefig(fig_name)
    plt.show()