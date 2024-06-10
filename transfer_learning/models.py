import torch
from torch import nn 

class TinyVGG(nn.Module):

  def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
      super().__init__()
      self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape, 
                    out_channels=hidden_units, 
                    kernel_size=3, 
                    stride=1, 
                    padding=0),  
          nn.ReLU(),
          nn.Conv2d(in_channels=hidden_units, 
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,
                        stride=2)
      )
      self.conv_block_2 = nn.Sequential(
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.MaxPool2d(2)
      )
      self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features=hidden_units*13*13,
                    out_features=output_shape)
      )

  def forward(self, x: torch.Tensor):
      x = self.conv_block_1(x)
      x = self.conv_block_2(x)
      x = self.classifier(x)
      return x
      # return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion


# 残差模块的搭建
class BasicBlock(nn.Module):
    def __init__(self,input_dim,output_dim,stride = [1,1],padding=1):
        super(BasicBlock,self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(input_dim,output_dim,kernel_size=3,stride=stride[0],padding=padding,bias=False),
                                 nn.BatchNorm2d(output_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(output_dim,output_dim,kernel_size=3,stride=stride[1],padding=padding,bias=False),
                                 nn.BatchNorm2d(output_dim))
        # shortcut 部分
    # 有可能出现维度不一致的情况，所以分情况进行处理
        self.shortcut = nn.Sequential()
        if stride[0] != 1 or input_dim != output_dim:
       # 解释一下，有些情况可能会出现输出和输入维度不一致的情况，就是咱们这个在进行残差连接的时候，出现维度不一致，
       # 这个时候就要通过大小为 1 的卷积核进行升维度，但是这个时候也要把图片进行下采样，所以stride = 2 这个时候刚好我们第一个卷积核的stride 也是2
            self.shortcut = nn.Sequential(nn.Conv2d(input_dim,output_dim,kernel_size=1,stride=stride[0],bias=False),
                                     nn.BatchNorm2d(output_dim))
       
    def forward(self,x):
       out = self.layer(x)
       out += self.shortcut(x)
       out = torch.relu(out)
       return out  

class Resnet18(nn.Module):
    def __init__(self,BasicBlock,num_class = 10):
        super(Resnet18,self).__init__()
        # 初始化input
        self.input_dim = 64
        # 输入 3*28*28
        self.conv1 = nn.Sequential(nn.Conv2d(3,64,kernel_size=3,padding=1,bias=False),  # 64*28*28
                                   nn.BatchNorm2d(64))  # 这里直接删除了下采样模块
        self.conv2 = self._make_layer(BasicBlock,64,[[1,1],[1,1]])
        self.conv3 = self._make_layer(BasicBlock,128,[[2,1],[1,1]])
        self.conv4 = self._make_layer(BasicBlock,256,[[2,1],[1,1]])
        self.conv5 = self._make_layer(BasicBlock,512,[[2,1],[1,1]])

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,num_class)
    def _make_layer(self,block,output_dim,strides):
        layer = []
        for stride in strides:
            layer.append(block(self.input_dim,output_dim,stride))
            # 下一层的input为这一层的output
            self.input_dim = output_dim
        return nn.Sequential(*layer)
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.avgpool(out)
        out = out.reshape(x.shape[0],-1)
        out = self.fc(out)
        return out



class AlexNet(nn.Module):
    def __init__(self):
        super().__init__() # 输入 3*28*28
        self.layer = nn.Sequential(nn.Conv2d(1,32,kernel_size=3,padding=1), #32*28*28
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(2,2),                    #32*14*14
                                   nn.Conv2d(32,64,kernel_size=3,padding=1),  # 64*14*14
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(2,2),                        # 64*7*7
                                   nn.Conv2d(64,128,kernel_size=3,padding=1),  # 128*7*7
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128,256,kernel_size=3,padding=1),    # 256*7*7
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256,256,kernel_size = 3,padding=1),    # 256*7*7
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(3,2))    # 256*3*3
        self.classifier = nn.Sequential(
            nn.Linear(256*3*3, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )
    
    def forward(self,x):
        x = self.layer(x)
        x = x.view(x.size(0), 256*3*3)
        x = self.classifier(x)
        return x