import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel,  BertModel
from torch_geometric.nn import GATConv
import copy
import torch.nn.functional as F

    
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一层
        self.relu = nn.ReLU()                       # ReLU 激活
        self.fc2 = nn.Linear(hidden_dim, output_dim) # 第二层

    def forward(self, x):
        """
        x 形状: [batch_size, input_dim]
        返回: [batch_size, output_dim]
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    
class TextNetwork(nn.Module):
    def __init__(self):
      
        super().__init__()
        self.dim = 768
        self.cross_attention_layer = nn.MultiheadAttention(
            embed_dim=self.dim, 
            num_heads=4, 
            batch_first=True
        )
   
        
        self.linear = nn.Linear(self.dim, 512)
    def forward(self, x):
        use_TQ = x['use_functions'] # B Class Functions 3 length
        B, C, F, D, dim = use_TQ.shape
      
        hx = torch.mean(use_TQ[:,:,:, [0, 1], :],dim = 3)  
        u =  use_TQ[:,:,:, 2, :]
      
        k_v = torch.mean(torch.stack([hx, u]),dim = 0)
        k_v = k_v.view(B*C, F, -1)
      
        use_cls = x['use_cls']
        B, C, F, D, dim = use_cls.shape
      
        T_x = use_cls
     
        T_x = torch.mean(T_x,dim=3) ## batch_size, classes, each_classes's functions, embedding_dim
        T_x = T_x.view(B*C,F, -1)
        
        T = self.cross_attention_layer(
            T_x,
            k_v,
            k_v
        )[0]
        T = T.view(B,C,F,-1)
        T = torch.mean(T, dim = 2) #B class embedding tim
        use_times = x['use_times'].float() #B Class
        # # print("T.shape",T.shape)
        use_times = torch.softmax(use_times,dim = 1)
        use_times = use_times.unsqueeze(-1)
        T = T * use_times
       
        T = torch.mean(T, dim = 1)
        T = self.linear(T)

        return T    
class AttentionMapClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AttentionMapClassifier, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 20 * 20, 512)  # 40x40 -> 20x20 -> 10x10 经过两次池化
        self.fc2 = nn.Linear(512, 512)   # 输出层
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(1024, 1)
        self.fc5 = nn.Linear(1024, 512)
        self.text_network = TextNetwork()

    def forward(self, x, good_edge,text):
        # 卷积 + 激活 + 池化
        x = torch.stack([x, good_edge], dim = 0)
        # x = good_edge.float() #wograph
        x = x.view(x.size(1) * 2,1,x.size(2),x.size(2))
        # x = x.view(x.size(0),1,x.size(2),x.size(2)) #wograph
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # 展平
        x = x.view(-1, 64 * 20 * 20)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(2, -1, 512)
        # x = x.view(1,-1,512) #wograph
        good_edge = x[1,:]
        x = x[0,:]
        # good_edge = x[0,:] #wograph
        cosine_similarity = F.cosine_similarity(x,good_edge, dim=1).view(x.size(0),1)
        t = torch.concat([x, good_edge],dim = 1)
        # print(t.shape)
        # t = torch.concat([good_edge, good_edge],dim = 1) #wograph
        alpha = torch.sigmoid(self.fc4(t))
        t = alpha * cosine_similarity
      
        zeros = torch.zeros(t.size(0),1).to(t.device)
        # t = torch.cat([zeros, t], dim = 1)
        t = torch.cat([t, zeros], dim = 1)
        
        a = self.text_network(text)
        
        y = self.fc5(torch.cat([a, x],dim=1))
         
        x = x + y  # Text Component
  
        x = self.fc3(x) #Graph Component
        
        x = x + t  # LLM component
        x = torch.softmax(x, dim = 1)
        return x



