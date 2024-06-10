import torch
import torch.nn as nn
from Config import *
import numpy as np
from create_source_dataset import *
from engine import *
import models
import matplotlib.pyplot as plt
from create_target_dataset import *
import matplotlib.pyplot as plt
"""图像的预处理和增强的相关的转换"""

# source_transform = transforms.Compose([
#     transforms.Resize(size=(28, 28)),
#     transforms.Grayscale(num_output_channels = 1),
#     transforms.RandomHorizontalFlip(p=0.5), 
#     transforms.RandomInvert(p=0.5), # 数据增强随机翻转
#     transforms.ToTensor() 
# ])

source_transform = transforms.Compose([
    transforms.Resize(size=(28, 28)),
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomInvert(p=0.5), # 数据增强随机翻转
    transforms.ToTensor() 
])



# ------------------对于目标数据集的处理---------------------------
target_transform = transforms.Compose([
    transforms.Resize(size=(28, 28)),
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.ToTensor() 
])


# target_transform = transforms.Compose([
#     transforms.Resize(size=(28, 28)),
#     transforms.Grayscale(num_output_channels = 1), #如果使用这个，记得改输入层
#     transforms.RandomHorizontalFlip(p=0.5), 
#     transforms.ToTensor() 
# ])







"""目标数据集的创建"""

target_train_dataset = mydataset(Config.train_x,Config.train_y,target_transform)
target_test_dataset = mydataset(Config.test_x,Config.test_y,target_transform)
target_train_dataloader = DataLoader(target_train_dataset,Config.batch_size,shuffle=True)
target_test_dataloader = DataLoader(target_test_dataset,Config.batch_size,shuffle=True)



"""源数据的创建"""

source_train_data = datasets.ImageFolder(root=Config.train_dir, 
                                  transform=source_transform, 
                                  target_transform=None) 
source_test_data = datasets.ImageFolder(root=Config.test_dir, 
                                 transform=source_transform)
source_train_dataloader = DataLoader(source_train_data,Config.batch_size,shuffle=True)
source_test_dataloader = DataLoader(source_test_data,Config.batch_size,shuffle=True)

if __name__ == '__main__':
    # # 定义模型
     #----------------------------------------------------下面是AlexNet-----------------------------------------------------#
    # model = models.AlexNet().to(Config.device)
    # train_loop(model,source_train_dataloader,source_test_dataloader,epochs = Config.epoch,evaluation=True)
    # ## ------------------------------------------下面是迁移学习的内容
    # # 读取在训练集上面训练好的参数
    # model_target = models.AlexNet().to(Config.device)
    # model_target.load_state_dict(torch.load(Config.save_path), strict=False)
    # # 冻结特征提取层的参数
    # for param in model_target.layer.parameters():
    #     param.requires_grad = False
    # # 由于输出两者一直,所以这里不重写分类层
    # train_loop(model_target,target_train_dataloader,target_test_dataloader,epochs = Config.epoch,evaluation=True)

    #----------------------------------------------------下面是Resnet18------------------------------------------------------#
    # 这种做法是 未对他俩的图像输入维度进行处理，都用的原始的图像
    model2 = models.Resnet18(BasicBlock=models.BasicBlock).to(Config.device)  # 初始化模型
    loss_source,time_source,acc_source = train_loop(model2,source_train_dataloader,Config.epoch_source,source_test_dataloader,evaluation=True) # 开始训练
    loss_draw(loss_source,time_source,acc_source,"source")
    
    # 初始化迁移模型，并读取参数
    model_target_2 = models.Resnet18(models.BasicBlock).to(Config.device)
    model_target_2.load_state_dict(torch.load(Config.save_path))
    # 重写网络层并进行参数冻结
    layers_to_freeze = [model_target_2.conv2,
                  model_target_2.conv3,
                  model_target_2.conv4,
                  model_target_2.conv5,]
    # 重写一下他的输入层，由于任务类似，就不重写输出层，仅仅训练参数即可
    model_target_2.conv1 =nn.Sequential(nn.Conv2d(3,64,kernel_size=3,padding=1,bias=False), 
                                   nn.BatchNorm2d(64)).to("cuda")
    for layer in layers_to_freeze:
        for param in layer.parameters():
            param.requires_grad = False    
    loss_target,time_target,acc_target = train_loop(model_target_2,target_train_dataloader,Config.epoch_target,target_test_dataloader,evaluation=True)
    loss_draw(loss_target,time_target,acc_target,"target")







    # 这种做法是 对源数据进行彩色话处理
    # model2 = models.Resnet18(BasicBlock=models.BasicBlock).to(Config.device)  # 初始化模型
    # loss_source,time_source,acc_source = train_loop(model2,source_train_dataloader,Config.epoch_source,source_test_dataloader,evaluation=True) # 开始训练
    # loss_draw(loss_source,time_source,acc_source,"source")
    
    # 初始化迁移模型，并读取参数
    # model_target_2 = models.Resnet18(models.BasicBlock).to(Config.device)
    # model_target_2.load_state_dict(torch.load(Config.save_path))
    # # 重写网络层并进行参数冻结
    # layers_to_freeze = [
    #               model_target_2.conv2,
    #               model_target_2.conv3,
    #               model_target_2.conv4,
    #               model_target_2.conv5,]
    # # 重写一下他的输入层，由于任务类似，就不重写输出层，仅仅训练参数即可
    # # model_target_2.conv1 =nn.Sequential(nn.Conv2d(3,64,kernel_size=3,padding=1,bias=False), 
    # #                                nn.BatchNorm2d(64)).to("cuda")
    # for layer in layers_to_freeze:
    #     for param in layer.parameters():
    #         param.requires_grad = False    
    # loss_target,time_target,acc_target = train_loop(model_target_2,target_train_dataloader,Config.epoch_target,target_test_dataloader,evaluation=True)
    # loss_draw(loss_target,time_target,acc_target,"target")
