import numpy as np
import torch
import network as net
import torch.nn as nn
esp = 1e-8
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from FPS import load_ply,farthest_point_sample,index_points,query_ball_point,relative_cordinate
from data_loader import PointCloudDataset,read_root_txt,index_to_points
from torch.utils.data import Dataset, DataLoader
from train_test import correct_num,Ratio
import matplotlib.pyplot as plt
from tqdm import tqdm





# 测试参数下，不同patchs 时测试集上的分类正确率
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_model = torch.load('new_params.pth',map_location='cpu')

Net=net.double_fusion()
Net.score_compute.load_state_dict(save_model)
Net =Net.to(device)

root_txt_path= './rank_pair_test.txt'
sample_path1_list, sample_path2_list, labels_list,distortion_type = read_root_txt(root_txt_path)
size = len(sample_path1_list)
Net.eval()
criterion = nn.BCELoss()
patch_number = [8]
# patch_number = [8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136]
#patch_number = [64]
repeat_number =10
total_patch = []
octree_patch = []
grid_patch = []
random_patch = []
noise1_patch = []
noise2_patch = []
noise3_patch = []
noise4_patch = []
pbar = tqdm(total=len(patch_number) * repeat_number * size)
for patchs in patch_number:
    total_accuracy = []
    octree_accuracy = []
    grid_accuracy = []
    random_accuracy = []
    noise1_accuracy = []
    noise2_accuracy = []
    noise3_accuracy = []
    noise4_accuracy = []
    for nums in range(repeat_number):
        accs2 = Ratio()
        loss2 = Ratio()  # 测试集
        octree_acc=  Ratio()
        grid_acc =  Ratio()
        random_acc =  Ratio()
        noise1_acc =  Ratio()
        noise2_acc =  Ratio()
        noise3_acc =  Ratio()
        noise4_acc =  Ratio()
        for index in range(size):
            pbar.update()
            path1 = sample_path1_list[index]
            path2 = sample_path2_list[index]  # 样本路径
            #label = labels_list[index]
            model1 = load_ply(path1).reshape(1, -1, 3)  # ply模型    BxNx3  B=1
            model2 = load_ply(path2).reshape(1, -1, 3)
            model1 = torch.from_numpy(model1)
            model2 = torch.from_numpy(model2)
            centroids_index = farthest_point_sample(model1, patchs)  # 每次采样点数
            centroids = index_points(model1, centroids_index)  # 确定采样中兴点坐标
            # radius采样
            result1 = query_ball_point(0.2, 516, model1, centroids)
            result2 = query_ball_point(0.2, 516, model2, centroids)  # B x S x nsample
            result1_np = result1.numpy()
            result2_np = result2.numpy()
            B, S, patch_size = result1_np.shape
            result1_value = np.zeros((B, S, patch_size, 3), dtype=float)
            result2_value = np.zeros((B, S, patch_size, 3), dtype=float)
            model1_numpy = model1.numpy()  # 此部分代码基于numpy运算，故转换
            for patch in range(S):
                patch_index = result1_np[:, patch, :]  # [B patch_size]
                value = index_to_points(model1_numpy, patch_index)  # [B patch_size C]
                for batch in range(B):
                    result1_value[batch][patch] = value[batch]  # B X S X patch_size X C
            model2_numpy = model2.numpy()  # 此部分代码基于numpy运算，故转换
            for patch in range(S):
                patch_index = result2_np[:, patch, :]  # [B patch_size]
                value = index_to_points(model2_numpy, patch_index)  # [B patch_size C]
                for batch in range(B):
                    result2_value[batch][patch] = value[batch]  # B X S X patch_size X C
            data1_tensor = torch.tensor(result1_value, dtype=torch.float)
            data2_tensor = torch.tensor(result2_value, dtype=torch.float)

            # 相对坐标转换
            data1_tensor = relative_cordinate(data1_tensor, centroids)
            data2_tensor = relative_cordinate(data2_tensor, centroids)
            # #data1_tensor = data1_tensor[0]
            # #data2_tensor = data2_tensor[0]  # S X patch_size X C
            # data_patch1=pc_normalize(data_patch1)
            # data_patch2=pc_normalize(data_patch2)  #   坐标零均值化
            #label_tensor = torch.tensor(label, dtype=torch.float)
            #label_tensor = label_tensor.unsqueeze(-1)
            # if self.category:
            #     return data1_tensor, data2_tensor, label_tensor, self.distortion_type[index]
            # else:
            #     return data1_tensor, data2_tensor, label_tensor
            data1 = data1_tensor.to(device)    # B x S X patch_size X C
            data2 = data2_tensor.to(device)
            #label = label_tensor.to(device)
            data1 = torch.transpose(data1, -1, -2)  # dataloader中数据为Bxrandom_sizexpatch_sizex3
            data2 = torch.transpose(data2, -1, -2)
            dist1, dist2, out = Net(data1, data2)
            num = correct_num(dist1, dist2)
            num = num.cpu()
            #loss_net = criterion(out, label)
            #loss2.update(loss_net.cpu().detach().item() * data1.size()[0], data1.size()[0])
            accs2.update(num, data1.size()[0])
            if distortion_type[index] == 'OctreeCom':
                octree_acc.update(num, data1.size()[0])
            elif distortion_type[index] == 'random':
                random_acc.update(num, data1.size()[0])
            elif distortion_type[index] == 'gridAverage':
               grid_acc.update(num, data1.size()[0])
            elif distortion_type[index] == 'noise1':
               noise1_acc.update(num, data1.size()[0])
            elif distortion_type[index] == 'noise2':
               noise2_acc.update(num, data1.size()[0])
            elif distortion_type[index] == 'noise3':
               noise3_acc.update(num, data1.size()[0])
            elif distortion_type[index] == 'noise4':
               noise4_acc.update(num, data1.size()[0])
        total_accuracy.append(accs2.ratio)
        octree_accuracy.append(octree_acc.ratio)
        grid_accuracy.append(grid_acc.ratio)
        random_accuracy.append(random_acc.ratio)
        noise1_accuracy.append(noise1_acc.ratio)
        noise2_accuracy.append(noise2_acc.ratio)
        noise3_accuracy.append(noise3_acc.ratio)
        noise4_accuracy.append(noise4_acc.ratio)
        print('\nAccuracy: {:.4f}%\n'.format(100. * accs2.ratio))
    total_patch.append(total_accuracy)
    octree_patch.append(octree_accuracy)
    grid_patch.append(grid_accuracy)
    random_patch.append(random_accuracy)
    noise1_patch.append(noise1_accuracy)
    noise2_patch.append(noise2_accuracy)
    noise3_patch.append(noise3_accuracy)
    noise4_patch.append(noise4_accuracy)
    if patchs == 64:
        print(np.mean(total_accuracy),np.std(total_accuracy))
        print(np.mean(octree_accuracy),np.std(octree_accuracy))
        print(np.mean(grid_accuracy),np.std(grid_accuracy))
        print(np.mean(random_accuracy),np.std(random_accuracy))
        print(np.mean(noise1_accuracy),np.std(noise1_accuracy))
        print(np.mean(noise2_accuracy),np.std(noise2_accuracy))
        print(np.mean(noise3_accuracy),np.std(noise3_accuracy))
        print(np.mean(noise4_accuracy),np.std(noise4_accuracy))

result = np.array([total_patch,octree_patch,grid_patch,random_patch,noise1_patch,noise2_patch,noise3_patch,noise4_patch])
np.save("accuracy_result_test_for_patchs.npy",result)

mean_total = np.mean(total_patch,-1)
mean_octree = np.mean(octree_patch,-1)
mean_grid = np.mean(grid_patch,-1)
mean_random = np.mean(random_patch,-1)
mean_noise1 = np.mean(noise1_patch,-1)
mean_noise2 = np.mean(noise2_patch,-1)
mean_noise3 = np.mean(noise3_patch,-1)
mean_noise4 = np.mean(noise4_patch,-1)
fig1 = plt.figure("accuracy")
plt.plot(patch_number, mean_total,'-o',label='Mean')
plt.plot(patch_number, mean_octree,'-s',label='GPCC')
plt.plot(patch_number, mean_grid,'-^',label='GD')
plt.plot(patch_number, mean_random,'-d',label='RS')
plt.plot(patch_number, mean_noise1,'-h',label='GN')
plt.plot(patch_number, mean_noise2,'-p',label='UN')
plt.plot(patch_number, mean_noise3,'-v',label='IN')
plt.plot(patch_number, mean_noise4,'-+',label='EN')
plt.xlabel("Patch Number")
plt.ylabel("accuracy")
plt.legend()
plt.savefig('accuracy_patch.jpg')
plt.show()

# result=np.load("accuracy_result_test_for_patchs.npy")
# total = result[0]
# octree = result[1]
# grid = result[2]
# random = result[3]
# noise1 = result[4]
# noise2 = result[5]
# noise3 = result[6]
# noise4 = result[7]
# print(np.mean(total,-1)[13],np.std(total,-1)[13])
# print(np.mean(octree,-1)[13],np.std(octree,-1)[13])
# print(np.mean(grid,-1)[13],np.std(grid,-1)[13])
# print(np.mean(random,-1)[13],np.std(random,-1)[13])
# print(np.mean(noise1,-1)[13],np.std(noise1,-1)[13])
# print(np.mean(noise2,-1)[13],np.std(noise2,-1)[13])
# print(np.mean(noise3,-1)[13],np.std(noise3,-1)[13])
# print(np.mean(noise4,-1)[13],np.std(noise4,-1)[13])
