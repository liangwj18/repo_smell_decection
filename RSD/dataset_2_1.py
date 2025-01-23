import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel,  BertModel
import numpy as np
import os
from tqdm import tqdm

class MyDataset(Dataset):
    def __init__(self, tasks, save_folder, model_name, batch, max_inherit_classes = 50, max_use_classes = 20, max_functions=10, max_len = 500, remake = False):
        """
        tasks: 列表，里面每个元素都是一个类似如下结构的字典
        {
            'inherit_forest': (Vertexs, Edge),
            'use_cls': [package, class],
            'cls_members_dict': {...},
            'use_members_statment': {...}
        }
        
        max_functions: 截断/归一化，每个类最多保留多少 methods
        max_statements: 截断/归一化，每个 method 最多保留多少 statements
        """
        self.tasks = tasks
        self.batch = batch
        self.max_len = max_len
        self.max_inherit_classes = max_inherit_classes
        self.max_use_classes = max_use_classes
        self.max_functions = max_functions
        self.model_name = model_name
 
        # self.max_statements = max_statements
  
        if os.path.exists(save_folder) == False:
            os.makedirs(save_folder)
        edge_path = os.path.join(save_folder, "edge.pth")
 
        self.n = 80
        if remake == True or os.path.exists(edge_path) == False:
            a = []
            n = self.n
            for task in self.tasks:
                Vertexs, Edge = task["data"]["inherit_forest"]
                attention_map = [[0 for i in range(n)] for j in range(n)]
                for i in range(n):
                    for j in range(n):
                        if i < len(Vertexs) and j < len(Vertexs) and Edge[i][j] == 1:
                            attention_map[i][j] = 1
                  
                attention_map = torch.FloatTensor(attention_map)
                label = torch.tensor(task['label'])
                a.append([attention_map, label])
            torch.save(a,edge_path)
            
        good_edge_path = os.path.join(save_folder,"good_edge.pth")
            
        self.edge = torch.load(edge_path)
        self.good_edge = torch.load(good_edge_path)

  
        tensor_path = os.path.join(save_folder, "tensor.pth")
        print(tensor_path)
        if remake == True or os.path.exists(tensor_path) == False:
            if os.path.exists(save_folder) == False:
                os.makedirs(save_folder)
            self.make_tensor(tensor_path)
        cls_path = os.path.join(save_folder, "cls.pth")
        if remake == True or os.path.exists(cls_path) == False:
            self.make_cls(tensor_path, cls_path)
           
        data = torch.load(cls_path)

        self.Vertexs_members = data['V']
        self.use_functions = data['functions']
        self.use_members = data['members']
        self.use_times = data['times']        
    
    def __len__(self):
        return len(self.edge)
        # return 100
  
   
    def get_cls(self, x, bertmodel):
        

        clsmodel = CLSModel(bertmodel)
        clsmodel = nn.DataParallel(clsmodel).cuda()
        B = 40 * 7
        N, C, F, D, L = x.shape
        x = x.view(-1,B, L)
        dataset = PreDataset(x)
       
        dataloader = DataLoader(dataset, batch_size=B, shuffle=False,num_workers = 4)
        x_list = []
        for _batch in tqdm(dataloader):
            batch, index = _batch
            batch = batch.cuda()
            attention_mask = torch.ones_like(batch).cuda()
            output, index = clsmodel(input_ids=batch, attention_mask = attention_mask, index = index)
            output_list = torch.unbind(output, dim=0)
            index_list = torch.unbind(index, dim = 0)
            for i in range(len(output_list)):
                x_list.append([output_list[i], index_list[i].item()])

        x_list.sort(key = lambda x:x[1])
        res_list = []
        for i in range(len(x_list)):
            res_list.append(x_list[i][0])
        x_list = torch.stack(res_list)
        x_cls = x_list.view(N,C,F,D,-1).cpu()
        return x_cls

    def __getitem__(self, index):
        """
        返回的数据格式示例：
         # output
        # function_x = x['inherit_functions'] #shape  batch_size, classes, each_classes's functions(cut),  2(text and return text), each_statement's length(cut 变定长了),  #Q is 0
        # edge_index = x['edge_index']
        # use_TQ = x['use_functions'] # B Class Functions 3 length
        # T_x = x['use_cls']
        # use_times = x['use_times'] #B Class
        """
     
       
        edge = self.edge[index]
        if index >= len(self.good_edge):
            good_edge = self.good_edge[-1].tolist()
        else:
            good_edge = self.good_edge[index].tolist()
        good_edge = pad_to_tensor(good_edge, (self.n, self.n),)
        
        use_members = self.use_members[index]
        use_functions = self.use_functions[index]
        use_times = self.use_times[index]
        return {"data":edge[0],"label":edge[1],"good_edge":good_edge,
                "text":{"use_functions":use_functions,
                "use_cls":use_members,
                "use_times":use_times  }}

       
    def make_tensor(self, tensor_path):
        self.V = []
        self.members = []
        self.functions = []
        self.times = []
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        for task_i in tqdm(range(len(self.tasks))):
            _task = self.tasks[task_i]
            task = _task['data']
            Vertexs, Edge = task["inherit_forest"]
            cls_members_dict = task["cls_members_dict"]
            Vertexs_members = [
                cls_members_dict[package+"!"+cls_name][1][:self.max_functions] for package, cls_name in Vertexs[:self.max_inherit_classes]
            ]
            for clss in Vertexs_members:
                for member_i, member in enumerate(clss):
                    member = member[1:]
                    for i, text in enumerate(member):
                        encoded = self.tokenizer(
                            text,
                            padding='max_length',
                            truncation=True,
                            max_length=self.max_len,
                            return_tensors='pt'    # 返回 pytorch 张量
                        )['input_ids'].squeeze(0).tolist()
                        member[i] = encoded
                    clss[member_i] = member
            inherit_class_num = self.max_inherit_classes if self.batch > 1 else min(self.max_inherit_classes, len(Vertexs_members))
            Vertexs_members = pad_to_tensor(Vertexs_members, (inherit_class_num, self.max_functions, 2, self.max_len),)
    
            # Vertexs_members = self.get_cls(Vertexs_members, bertmodel)
        
            # 2. 取出 use_cls
            use_cls = task["use_cls"][:self.max_use_classes]  # [package, class]
            use_members = [
                cls_members_dict[package+"!"+cls_name][1][:self.max_functions] for package, cls_name in use_cls
            ]
            for clss in use_members:
                for member_i, member in enumerate(clss):
                    member = member[1:]
                    for i, text in enumerate(member):
                        encoded = self.tokenizer(
                            text,
                            padding='max_length',
                            truncation=True,
                            max_length=self.max_len,
                            return_tensors='pt'    # 返回 pytorch 张量
                        )['input_ids'].squeeze(0).tolist()
                        member[i] = encoded
                    clss[member_i] = member

            use_functions = []
            use_times = []
            use_members_statment = task["use_members_statment"]  # { (pkg, cls): [ [...], [...], ... ] }
            for i in range(len(use_cls[1:])):
                use_functions.append([])
                cnt = 0
                for k in range(2):
                    if k == 0:
                        members = use_members[i+1]
                    else:
                        members = use_members[0]
                    for j in range(len(members)):
                        if j in use_members_statment[i+1][k]:
                            cnt += 1
                            text = use_members_statment[i+1][k][j]
                            encoded = self.tokenizer(
                                text,
                                padding='max_length',
                                truncation=True,
                                max_length=self.max_len,
                                return_tensors='pt'    # 返回 pytorch 张量
                            )['input_ids'].squeeze(0).tolist()
                            use_functions[i].append(
                                [members[j][0], members[j][1], encoded]
                            )
                use_times.append(cnt)
            use_class_num = self.max_use_classes if self.batch > 1 else min(self.max_use_classes, len(use_members))
            use_members = pad_to_tensor(use_members, (use_class_num, self.max_functions, 2, self.max_len))
            use_functions = pad_to_tensor(use_functions, (use_class_num, self.max_functions, 3, self.max_len))
    
            use_times = pad_to_tensor(use_times, (use_class_num,))
            self.V.append(Vertexs_members)
            self.functions.append(use_functions)
            self.members.append(use_members)
            self.times.append(use_times)
        
            if task_i % 500 == 0:
                torch.save({
                    "V":self.V,
                    "functions":self.functions,
                    "members":self.members,
                    "times":self.times
                },save_path)
        
        torch.save({
            "V":self.V,
            "functions":self.functions,
            "members":self.members,
            "times":self.times
        },tensor_path)
    def make_cls(self, tensor_path, cls_path):
        data = torch.load(tensor_path)
        self.V = data['V']
        self.functions = data['functions']
        self.members = data['members']
        self.times = data['times']
        bertmodel = BertModel.from_pretrained(self.model_name)    
        for param in bertmodel.parameters():
            param.requires_grad = False
        bertmodel = bertmodel.to("cuda")
        res = torch.concat([torch.stack(self.V,dim = 0), torch.stack(self.members, dim = 0), torch.stack(self.functions, dim = 0)],dim = -2)
        res = self.get_cls(res, bertmodel)
        self.V = res[:,:,:,0:2,:]
        self.members = res[:,:,:,2:4,:]
        self.functions = res[:,:,:,4:,:]
        
        torch.save({
            "V":self.V,
            "functions":self.functions,
            "members":self.members,
            "times":self.times
        },cls_path)
        

def create_filled_list(shape, fill_value=0):
    """
   
    """
    if len(shape) == 0:
      
        return fill_value

    return [create_filled_list(shape[1:], fill_value) for _ in range(shape[0])]


def pad_nested_list(data, shape, fill_value=0):
   

    if len(shape) == 0:
        return data  
    

    target_len = shape[0]
    
    if not isinstance(data, list):
        data = [data]
    
    if len(data) > target_len:
        print(data)
        print(shape)
        raise ValueError()
    
    padded_layer = []
    for i in range(len(data)):
        if len(shape) > 1:
            padded_layer.append(
                pad_nested_list(data[i], shape[1:], fill_value=fill_value)
            )
        else:
            padded_layer.append(data[i])
    
    for _ in range(target_len - len(data)):
        if len(shape) > 1:
            padded_layer.append(create_filled_list(shape[1:], fill_value))
        else:
            padded_layer.append(fill_value)
    
    return padded_layer


def pad_to_tensor(data, shape, fill_value=0):
    """
    综合函数: 对嵌套不规则列表 data 补齐到形状 shape，最后转换为 PyTorch Tensor。
    """
    padded_list = pad_nested_list(data, shape, fill_value)
    return torch.tensor(padded_list)


class PreDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data.view(-1, data.size(-1))
        self.index = torch.LongTensor(torch.arange(self.data.size(0)))
        print(self.index)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.index[idx]

class CLSModel(nn.Module):
    def __init__(self, bertmodel):
        super(CLSModel, self).__init__()
        self.bertmodel = bertmodel
    def forward(self, input_ids, attention_mask, index):
        res = self.bertmodel(input_ids = input_ids, attention_mask = attention_mask).pooler_output
        return res, index