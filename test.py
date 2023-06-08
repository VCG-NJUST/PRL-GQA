'''
做一些代码测试用，与正式模型训练无关
'''

import h5py

f = h5py.File('ply_data_train2.h5','r')   #打开h5文件
data =f['data']
print(data)
# path='point.txt'
# f_ = open(path, 'w', encoding='utf-8')
# for i in range(0,2048):
#     f_.write(str(data[200][i][0])+' ')
#     f_.write(str(data[200][i][1])+' ')
#     f_.write(str(data[200][i][2]))
#     f_.write('\n')
# import cv2
# image_A = cv2.imread("ref_1.png")
# print (image_A.shape)
# import numpy as np
# a=np.arange(0,30,5)
# b=np.arange(0,30,5)
# print(np.concatenate((a,b),axis=0))
import torch
import torch.optim as optim
from data_loader import PointCloudDataset
from torch.utils.data import Dataset, DataLoader
import network as net
import torch.nn as nn
import os
import general
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(torch.__version__)

train_Dataset = PointCloudDataset('./rank_pair_train.txt')
test_Dataset = PointCloudDataset('./rank_pair_test.txt')
test_path='./G-PCD/stimuli'
out_score='./G-PCD/test_score.txt'
trainloader = DataLoader(train_Dataset, batch_size=8, num_workers=0, shuffle=True,
                             drop_last=False)
testloader = DataLoader(test_Dataset, batch_size=1, num_workers=0, shuffle=False,
                             drop_last=False)

Net= net.BasicFCModule(308, 1)
Net.load_state_dict(torch.load('./backupdata/params.pth'))
criterion = nn.BCELoss()
#net.test(testloader,Net,criterion,epoch=1,mode=1)
#net.test(trainloader,Net,criterion,epoch=1,mode=0)

test_data=os.listdir(test_path)
score_txt = open(out_score, mode="w", encoding="utf-8")
VFH=[]
for i in test_data:
    if i.endswith("VFH.txt"):
        VFH.append(i)
# for i in range(8,45,9):
#     VFH.insert(i-8,VFH[i])
#     VFH.pop(i+1)
x=general.read_sample_txt_files(test_path+'/'+'bunny_VFH.txt')
x=torch.tensor(x,dtype=torch.float)
x=x.unsqueeze(0)
print(len(VFH))
Net.eval()
for i in VFH:
    path=test_path+'/'+i
    x1=torch.tensor(general.read_sample_txt_files(path),dtype=torch.float)
    x1=x1.unsqueeze(0)
    score,_,_=Net.forward(x1,x)
    score_txt.write(str(score.item())+'\n')
# x1=torch.tensor(general.read_sample_txt_files(test_path+'/'+'bunny_D02_L04_VFH.txt'),dtype=torch.float)
# x1=x1.unsqueeze(0)
# print(x1.size())