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
from tqdm import tqdm

'''
@author leon
@desc 直接将排序的一条分支用于GPCD数据集测试,可视化并保存采样不同patch数下的结果
@date 2022/1
'''

data = pd.read_csv("../fine_turn/GPCD/G-PCD/subjective scores/desktop setup/subj_desktop_dsis.csv")  # 主观分数
data_dict = {}
for row in range(data.shape[0]):
    if (data.iloc[row, 0].endswith('hidden')):
        data_dict[data.iloc[row, 0][0:-11]] = data.iloc[row, 1:].mean()
    else:
        data_dict[data.iloc[row, 0][:]] = data.iloc[row, 1:].mean()
# save_model = torch.load('../new_params.pth')
save_model = torch.load('new_params_lz.pth')
score_Net = net.weight_score()
score_Net.load_state_dict(save_model)
test_path = '../fine_turn/GPCD/G-PCD/stimuli'  # 失真模型
model_list = os.listdir(test_path)
# patch_number = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160]
patch_number = [112]
PLCC_val = []
SRCC_val = []
KRCC_val = []
RMSE_val = []
repeat_number = 10
pbar = tqdm(total=len(patch_number) * repeat_number * len(model_list))
for num in patch_number:
    PLCC_accu = []
    SRCC_accu = []
    KRCC_accu = []
    RMSE_accu = []
    for k in range(repeat_number):  # 重复十次取平均
        score_dict = {}
        for i in model_list:
            pbar.update(1)
            if i.endswith('.ply'):
                path = test_path + '/' + i
                model = fps.load_ply(path).reshape(1, -1, 3)  # ply模型    BxNx3  B=1
                model = torch.from_numpy(model)
                centroids_index = fps.farthest_point_sample(model, num)  # 测试采样点数
                centroids = fps.index_points(model, centroids_index)
                # radius采样
                result1 = fps.query_ball_point(0.2, 516, model, centroids)
                result1_np = result1.numpy()
                B, S, patch_size = result1_np.shape
                result1_value = np.zeros((B, S, patch_size, 3), dtype=float)
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
                score = score_Net(data_tensor_trans)  # B X 1
                score = score[0]
                score_dict[i[0:-4]] = score.item()
        predict_MOS = np.zeros((len(score_dict), 2), dtype=float)
        key_list = list(score_dict.keys())
        for i in range(len(score_dict)):
            predict_MOS[i] = np.asarray([score_dict[key_list[i]], data_dict[key_list[i]]])
        x = predict_MOS[:, 0]
        y = predict_MOS[:, 1]
        # GPCD_single_branch = open('./single_branch_for_GPCD.txt', mode="w", encoding="utf-8")
        # for i in range(len(x)):
        #      GPCD_single_branch.write(str(x[i])+' ')
        # GPCD_single_branch.write('\n')
        # for i in range(len(x)):
        #      GPCD_single_branch.write(str(y[i])+' ')
        # GPCD_single_branch.write('\n')
        PLCC, SRCC, KRCC, RMSE = corr_value(y, x)
        # print('PLCC :{0},SRCC:{1},KRCC:{2},RMSE:{3}'.format(PLCC,SRCC,KRCC,RMSE))
        # GPCD_single_branch.close()
        PLCC_accu.append(PLCC)
        SRCC_accu.append(SRCC)
        KRCC_accu.append(KRCC)
        RMSE_accu.append(RMSE)
    PLCC_val.append(PLCC_accu)
    SRCC_val.append(SRCC_accu)
    KRCC_val.append(KRCC_accu)
    RMSE_val.append(RMSE_accu)

np.save('PLCC_for_patch_number.npy', np.array(PLCC_val))
np.save('SRCC_for_patch_number.npy', np.array(SRCC_val))
np.save('KRCC_for_patch_number.npy', np.array(KRCC_val))
np.save('RMSE_for_patch_number.npy', np.array(RMSE_val))

print('PLCC_val', PLCC_val)
print('SRCC_val', SRCC_val)
print('KRCC_val', KRCC_val)
print('RMSE_val', RMSE_val)

print('PLCC_val mean', np.mean(PLCC_val))
print('SRCC_val mean', np.mean(SRCC_val))
print('KRCC_val mean', np.mean(KRCC_val))
print('RMSE_val mean', np.mean(RMSE_val))
# fig1 = plt.figure("Coefficient ")
# plt.plot(patch_number, np.mean(PLCC_val, -1), '-o', label='PLCC')
# plt.plot(patch_number, np.mean(SRCC_val, -1), '-s', label='SRCC')
# plt.plot(patch_number, np.mean(KRCC_val, -1), '-^', label='KRCC')
# # plt.plot(patch_number, RMSE_val,'-p',label='RMSE')
# plt.xlabel("Patch Number")
# plt.ylabel("Value")
# plt.legend()
# plt.savefig('result_newfea.jpg')
# plt.show()

# PLCC_val = np.load('PLCC_for_patch_number.npy')
# SRCC_val = np.load('SRCC_for_patch_number.npy')
# KRCC_val = np.load('KRCC_for_patch_number.npy')
# RMSE_val = np.load('RMSE_for_patch_number.npy')

# fig1 = plt.figure("Coefficient")
# plt.plot(patch_number, np.mean(PLCC_val,-1),'-o',label='PLCC')
# plt.plot(patch_number, np.mean(SRCC_val,-1),'-s',label='SRCC')
# plt.plot(patch_number, np.mean(KRCC_val,-1),'-^',label='KRCC')
# plt.plot(patch_number, np.mean(RMSE_val,-1),'-p',label='RMSE')
# plt.xlabel("Patch Number")
# plt.ylabel("Value")
# plt.legend()
# plt.savefig('result_newfea.jpg',dpi=300)
# plt.show()
# print(np.mean(PLCC_val,-1)[9],np.std(PLCC_val,-1)[9])
# print(np.mean(SRCC_val,-1)[9],np.std(SRCC_val,-1)[9])
# print(np.mean(KRCC_val,-1)[9],np.std(KRCC_val,-1)[9])
# print(np.mean(RMSE_val,-1)[9],np.std(RMSE_val,-1)[9])


# 多次随机采样下系数值的变化情况
# data = pd.read_csv("../fine_turn/GPCD/G-PCD/subjective scores/desktop setup/subj_desktop_dsis.csv")   # 主观分数
# data_dict={}
# for row in range(data.shape[0]):
#     if(data.iloc[row,0].endswith('hidden')):
#         data_dict[data.iloc[row,0][0:-11]]=data.iloc[row,1:].mean()
#     else:
#         data_dict[data.iloc[row,0][:]]=data.iloc[row,1:].mean()
# save_model = torch.load('../params.pth')
# score_Net=net.weight_score()
# score_Net.load_state_dict(save_model)
# test_path='../fine_turn/GPCD/G-PCD/stimuli'    # 失真模型
# model_list = os.listdir(test_path)
# patch_number = 64
# repeat_number = 100
# PLCC_accu = []
# SRCC_accu = []
# KRCC_accu = []
# RMSE_accu = []
# for k in range(repeat_number):  # 重复十次取平均
#     score_dict = {}
#     for i in model_list:
#         if i.endswith('.ply'):
#             path=test_path+'/'+i
#             model = fps.load_ply(path).reshape(1, -1, 3)  # ply模型    BxNx3  B=1
#             model = torch.from_numpy(model)
#             centroids_index=fps.farthest_point_sample(model,patch_number)  # 测试采样点数
#             centroids = fps.index_points(model,centroids_index)
#             # radius采样
#             result1 = fps.query_ball_point(0.2, 516, model, centroids)
#             result1_np = result1.numpy()
#             B, S, patch_size = result1_np.shape
#             result1_value = np.zeros((B, S, patch_size, 3), dtype=float)
#             model1_numpy = model.numpy()  # 此部分代码基于numpy运算，故转换
#             for patch in range(S):
#                 patch_index = result1_np[:, patch, :]  # [B patch_size]
#                 value = index_to_points(model1_numpy, patch_index)  # [B patch_size C]
#                 for batch in range(B):
#                     result1_value[batch][patch] = value[batch]  # B X S X patch_size X C
#             data1_tensor = torch.tensor(result1_value, dtype=torch.float)
#             # 相对坐标转换
#             data1_tensor = fps.relative_cordinate(data1_tensor, centroids)
#             data_tensor_trans = torch.transpose(data1_tensor, -1, -2)     #B X S x C X patch_size
#             score = score_Net(data_tensor_trans)   # B X 1
#             score = score[0]
#             score_dict[i[0:-4]] = score.item()
#     predict_MOS = np.zeros((len(score_dict),2), dtype=float)
#     key_list = list(score_dict.keys())
#     for i in range(len(score_dict)):
#         predict_MOS[i] = np.asarray([score_dict[key_list[i]],data_dict[key_list[i]]])
#     x = predict_MOS[:,0]
#     y = predict_MOS[:,1]
#     # GPCD_single_branch = open('./single_branch_for_GPCD.txt', mode="w", encoding="utf-8")
#     # for i in range(len(x)):
#     #      GPCD_single_branch.write(str(x[i])+' ')
#     # GPCD_single_branch.write('\n')
#     # for i in range(len(x)):
#     #      GPCD_single_branch.write(str(y[i])+' ')
#     # GPCD_single_branch.write('\n')
#     PLCC,SRCC,KRCC,RMSE = corr_value(y,x)
#     # print('PLCC :{0},SRCC:{1},KRCC:{2},RMSE:{3}'.format(PLCC,SRCC,KRCC,RMSE))
#     # GPCD_single_branch.close()
#     PLCC_accu.append(PLCC)
#     SRCC_accu.append(SRCC)
#     KRCC_accu.append(KRCC)
#     RMSE_accu.append(RMSE)
#
# np.save('PLCC_for_repeat.npy',np.array(PLCC_accu))
# np.save('SRCC_for_repeat.npy',np.array(SRCC_accu))
# np.save('KRCC_for_repeat.npy',np.array(KRCC_accu))
# np.save('RMSE_for_repeat.npy',np.array(RMSE_accu))
# fig1 = plt.figure("Coefficient")
# x=range(1,len(PLCC_accu)+1)
# plt.plot(x, PLCC_accu,'-o',label='PLCC')
# plt.plot(x, SRCC_accu,'-s',label='SRCC')
# plt.plot(x, KRCC_accu,'-^',label='KRCC')
# #plt.plot(patch_number, RMSE_val,'-p',label='RMSE')
# plt.xlabel("repeat number")
# plt.ylabel("Value")
# plt.legend()
# plt.savefig('stability.jpg')
# plt.show()
