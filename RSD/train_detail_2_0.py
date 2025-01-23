from utils_code.utils_0 import useful_design_smell, read_task_in_jsonl, output_jsonl, smell_description_dic
from .network import AttentionMapClassifier
from .dataset_2_1 import MyDataset

from torch.utils.data import Dataset, DataLoader, SequentialSampler

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import os
import json

import argparse


def train(args):
   
    model = AttentionMapClassifier(num_classes = 2)
   
    model.cuda()
    

    tasks = read_task_in_jsonl(args.task_file_path)
    # tasks = None
    save_folder_name = "_".join(args.task_file_path.split("/")[-2:]).split(".jsonl")[0]

    pretreatment_save_folder = f"./RSD/RSD_pretreatment/{save_folder_name}"
    train_dataset = MyDataset(tasks, pretreatment_save_folder, args.model_name_or_path,batch = args.batch_size, max_inherit_classes = 20, max_use_classes = 20,  max_functions=10,max_len = 200, remake = False)
    
    sampler = SequentialSampler(train_dataset)
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
  
        sampler = sampler,
        num_workers = 4
    )
  
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    rank = 0
    model.train()
    num_epochs = args.epoch 
    step_count = 0
    train_loss_list = []
    train_acc_list = []
    criterion = nn.CrossEntropyLoss().cuda(rank)
    exp_name = args.exp_name
    save_folder = f"./RSD/train_model/{exp_name}/{save_folder_name}"
    if os.path.exists(save_folder) == False:
        os.makedirs(save_folder)
    if rank == 0:
        writer = SummaryWriter(log_dir=save_folder)
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch in tqdm(dataloader):
            step_count += 1

            data = batch["data"].cuda(rank)
            good_edge = batch["good_edge"].cuda(rank)
            text = batch["text"]
            for key in text:
                text[key] = text[key].cuda(rank)
          
            labels = batch["label"].cuda(rank)

            prediction = model(
                x = data,
                good_edge = good_edge,
                text = text
            )
       
            loss = criterion(prediction, labels)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(prediction, dim=1) 
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()
        

         
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        train_loss_list.append(epoch_loss)
        train_acc_list.append(epoch_acc)
        if rank == 0:
        
            writer.add_scalar("Loss/train", epoch_loss, epoch)
            writer.add_scalar("ACC/train", epoch_acc, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    
        if epoch % 5 == 0 or epoch + 1 == num_epochs:
            save_path = os.path.join(save_folder,"checkpoint.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "train_loss_list": train_loss_list,
                "train_acc_list": train_acc_list
            }, save_path)
    if rank == 0:
        writer.close()
   
    
    
def test(args):
    
    model = AttentionMapClassifier(num_classes = 2)
   
  
    model.cuda()
    model.eval()
   
    test_file_path = args.task_file_path
    tasks = read_task_in_jsonl(test_file_path)
    # tasks = None
    save_folder_name = "_".join(test_file_path.split("/")[-2:]).split(".jsonl")[0]
    assert save_folder_name.find("train") == -1
    exp_name = args.exp_name
    save_folder = f"./RSD/train_model/{args.exp_name}/{save_folder_name}"
    pretreatment_save_folder = f"./RSD/RSD_pretreatment/{save_folder_name}"
    test_dataset = MyDataset(tasks, pretreatment_save_folder, args.model_name_or_path,batch = args.batch_size, max_inherit_classes = 20, max_use_classes = 20,  max_functions=10,max_len = 200)
    # return 
    
    save_path = os.path.join(save_folder,"checkpoint.pth")
    weights_dict = torch.load(save_path.replace("test","train"), map_location='cpu')
    model.load_state_dict(weights_dict['model_state_dict'], strict=False)
    

    sampler = SequentialSampler(test_dataset)
    dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
  
        sampler = sampler,
        num_workers = 4
    )
  
   
    rank = 0
   
    num_epochs = 1 
    step_count = 0
    train_loss_list = []
    train_acc_list = []
   
    x = []
    y = []
    for epoch in range(num_epochs):
    
        correct = 0
        total = 0
        for batch in tqdm(dataloader):
            step_count += 1

            data = batch["data"].cuda(rank)
            good_edge = batch["good_edge"].cuda(rank)
          
            labels = batch["label"].cuda(rank)
            text = batch["text"]
            for key in text:
                text[key] = text[key].cuda(rank)

       
            prediction = model(
                x = data,
                good_edge = good_edge,
                text = text
            )
          
            
            _, preds = torch.max(prediction, dim=1)
            correct += (preds == labels).sum().item()
            x += preds.tolist()
            y += labels.tolist()
            total += labels.size(0)
          
        

        epoch_acc = correct / total
        
        
        train_acc_list.append(epoch_acc)
     
        print(f"Epoch [{epoch+1}/{num_epochs}] -  Accuracy: {epoch_acc:.4f}")
    if os.path.exists(save_folder) == False:
        os.makedirs(save_folder)
 
    test_file_path = os.path.join(save_folder, "test_file.json")
    json.dump({"labels":y,"preds":x},open(test_file_path,'w'))
      
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smell_type", type=str)
    parser.add_argument("--task_file_path", type = str)
    parser.add_argument("--model_name_or_path", type = str)
    parser.add_argument("--adapter_name_or_path", default = None)
    parser.add_argument("--output_folder",type=str)
    parser.add_argument("--batch_size", default = 10)
    parser.add_argument("--lr", default = 5e-6)
    parser.add_argument("--epoch", default = 200)
    parser.add_argument("--port")
    parser.add_argument("--benchmark")
    parser.add_argument("--exp_name")
    args = parser.parse_args()
   
    if args.benchmark == "train":
        train(args)
        # pass
    else:
        test(args)
