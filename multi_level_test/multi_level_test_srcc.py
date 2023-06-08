import torch
import network as net
import torch.nn as nn

esp = 1e-8
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
import pandas as pd
import FPS as fps
import numpy as np
from data_loader import index_to_points
from fine_turn.coefficient_calu import corr_value

'''
@author leon
@desc 
@date 2022/1
'''





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_model = torch.load('../new_params.pth')
score_Net=net.weight_score()
score_Net.load_state_dict(save_model)
score_Net =score_Net.to(device)
test_path='./bunny'    # 失真模型

model_list = os.listdir(test_path)
patch_number = [112]
MOS = np.arange(1,31,1)
MOS =np.array(MOS,dtype=np.float)
repeat_number = 10
path_list = os.listdir(test_path)
orignial_model = ' '
for file in path_list:
    path = test_path+'/'+file
    if os.path.isfile(path) and file.endswith('.ply'):
        orignial_model=path
distortion_score_dic= {}
distortion_SRCC_dic = {}
for file in path_list:
    dir_path= test_path + '/' + file
    if os.path.isdir(dir_path):
        distortion_name = file
        patch_repeat_score = []
        patch_repeat_SRCC = []
        for num in patch_number:
            repeat_score= []
            SRCC_list = []
            for k in range(repeat_number):
                score_list =[]
                for val in range(1,31):
                    model_path = dir_path + '/' + 'level' + str(val)+'.ply'
                    model = fps.load_ply(model_path).reshape(1, -1, 3)  # ply模型    BxNx3  B=1
                    model = torch.from_numpy(model)
                    centroids_index = fps.farthest_point_sample(model, num)  # 测试采样点数
                    centroids = fps.index_points(model, centroids_index)
                    # radius采样
                    result1 = fps.query_ball_point(0.2, 516, model, centroids)
                    result1_np = result1.numpy()
                    B, S, patch_size = result1_np.shape
                    result1_value = np.zeros((B, S, patch_size, 3), dtype=np.float)
                    model1_numpy = model.numpy()  # 此部分代码基于numpy运算，故转换
                    for patch in range(S):
                        patch_index = result1_np[:, patch, :]  # [B patch_size]
                        value = index_to_points(model1_numpy, patch_index)  # [B patch_size C]
                        for batch in range(B):
                            result1_value[batch][patch] = value[batch]  # B X S X patch_size X C
                    data1_tensor = torch.tensor(result1_value, dtype=torch.float)
                    # 相对坐标转换
                    data1_tensor = fps.relative_cordinate(data1_tensor, centroids)
                    data_tensor_trans = torch.transpose(data1_tensor, -1, -2)  # B X S x C X patch_size
                    data_tensor_trans = data_tensor_trans.to(device)
                    score = score_Net(data_tensor_trans)  # B X 1
                    score = score[0]
                    score_list.append(score)
                score_list = np.array(score_list,dtype =np.float)
                _, SRCC, _, _ = corr_value(MOS,score_list)
                SRCC_list.append(SRCC)
                repeat_score.append(score_list)
            patch_repeat_score.append(repeat_score)
            patch_repeat_SRCC.append(SRCC_list)
        distortion_score_dic[distortion_name]= patch_repeat_score
        distortion_SRCC_dic[distortion_name] = patch_repeat_SRCC
np.save('result_srcc_dic.npy',distortion_SRCC_dic)
np.save('score_dic.npy',distortion_score_dic)
# distortion_SRCC_dic=np.load('result_dic.npy',allow_pickle=True).item()
# print(distortion_SRCC_dic)
# for (key,array) in distortion_score_dic.items():
#     score1 = array[0][0]
#     score2 =array[0][1]
#     print(key + ':\n')
#     print('score1: ')
#     print(score1)
#     print('\nscore2:')
#     print(score2)

for (key,array )in distortion_SRCC_dic.items():
    srocc_ar = array[0]

    srocc_mean = np.mean(np.array(srocc_ar))
    srocc_std = np.std(np.array(srocc_ar))
    print(key + ':\n')
    print('mean: {:.4f}  ,std:   {:.4f}  \n '.format(srocc_mean,srocc_std))







