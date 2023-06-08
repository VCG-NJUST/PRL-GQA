import numpy as np
import torch
import network as net
import torch.nn as nn

esp = 1e-8
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from data_loader import PointCloudDataset
from torch.utils.data import Dataset, DataLoader
from train_test import correct_num, Ratio

# 测试参数下，直接计算训练集与数据集上的分类正确率与损失
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# save_model = torch.load('params.pth',map_location='cpu')
save_model = torch.load('new_params.pth', map_location='cpu')

Net = net.double_fusion()
Net.score_compute.load_state_dict(save_model)
Net = Net.to(device)
train_Dataset = PointCloudDataset('./rank_pair_train.txt', True)
test_Dataset = PointCloudDataset('./rank_pair_test.txt', True, 112)
trainloader = DataLoader(train_Dataset, batch_size=1, num_workers=0, shuffle=True,
                         drop_last=False)
testloader = DataLoader(test_Dataset, batch_size=1, num_workers=0, shuffle=False,
                        drop_last=False)
Net.eval()
criterion = nn.BCELoss()
accs1 = Ratio()
loss1 = Ratio()  # 训练集
accs_comandown1 = Ratio()
accs_noise1 = Ratio()

# for idx, (data1, data2, label,category) in enumerate(trainloader):
#     data1 = data1.to(device)
#     data2 = data2.to(device)
#     label = label.to(device)
#     data1 = torch.transpose(data1, -1, -2)  # dataloader中数据为Bxrandom_sizexpatch_sizex3
#     data2 = torch.transpose(data2, -1, -2)
#     dist1, dist2, out = Net(data1, data2)
#     num = correct_num(dist1, dist2)
#     num = num.cpu()
#     loss_net = criterion(out, label)
#     loss1.update(loss_net.cpu().detach().item() * data1.size()[0], data1.size()[0])
#     accs1.update(num, data1.size()[0])
#     if category[0] == 'com&down':
#         accs_comandown1.update(num, data1.size()[0])
#     else:
#         accs_noise1.update(num, data1.size()[0])
#
# print('\ntrain set: Average loss: {:.4f}, Accuracy: {:.4f}%\n'.format(
#     loss1.ratio, 100. * accs1.ratio))
# print('com & down Accuracy: {:.4f}%, noise Accuracy: {:.4f}%\n'.format(
#             100. * accs_comandown1.ratio, 100. * accs_noise1.ratio))


repeat_number = 10
total_accuracy = []
octree_accuracy = []
grid_accuracy = []
random_accuracy = []
noise1_accuracy = []
noise2_accuracy = []
noise3_accuracy = []
noise4_accuracy = []

for num in range(repeat_number):
    accs2 = Ratio()
    loss2 = Ratio()  # 测试集
    octree_acc = Ratio()
    grid_acc = Ratio()
    random_acc = Ratio()
    noise1_acc = Ratio()
    noise2_acc = Ratio()
    noise3_acc = Ratio()
    noise4_acc = Ratio()
    for idx, (data1, data2, label, category) in enumerate(testloader):
        data1 = data1.to(device)
        data2 = data2.to(device)
        label = label.to(device)
        data1 = torch.transpose(data1, -1, -2)  # dataloader中数据为Bxrandom_sizexpatch_sizex3
        data2 = torch.transpose(data2, -1, -2)
        dist1, dist2, out = Net(data1, data2)
        num = correct_num(dist1, dist2)
        num = num.cpu()
        loss_net = criterion(out, label)
        loss2.update(loss_net.cpu().detach().item() * data1.size()[0], data1.size()[0])
        accs2.update(num, data1.size()[0])
        if category[0] == 'OctreeCom':
            octree_acc.update(num, data1.size()[0])
        elif category[0] == 'random':
            random_acc.update(num, data1.size()[0])
        elif category[0] == 'gridAverage':
            grid_acc.update(num, data1.size()[0])
        elif category[0] == 'noise1':
            noise1_acc.update(num, data1.size()[0])
        elif category[0] == 'noise2':
            noise2_acc.update(num, data1.size()[0])
        elif category[0] == 'noise3':
            noise3_acc.update(num, data1.size()[0])
        elif category[0] == 'noise4':
            noise4_acc.update(num, data1.size()[0])

    total_accuracy.append(accs2.ratio)
    octree_accuracy.append(octree_acc.ratio)
    grid_accuracy.append(grid_acc.ratio)
    random_accuracy.append(random_acc.ratio)
    noise1_accuracy.append(noise1_acc.ratio)
    noise2_accuracy.append(noise2_acc.ratio)
    noise3_accuracy.append(noise3_acc.ratio)
    noise4_accuracy.append(noise4_acc.ratio)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.4f}%\n'.format(
        loss2.ratio, 100. * accs2.ratio))
print(np.mean(total_accuracy), np.std(total_accuracy))
print(np.mean(octree_accuracy), np.std(octree_accuracy))
print(np.mean(grid_accuracy), np.std(grid_accuracy))
print(np.mean(random_accuracy), np.std(random_accuracy))
print(np.mean(noise1_accuracy), np.std(noise1_accuracy))
print(np.mean(noise2_accuracy), np.std(noise2_accuracy))
print(np.mean(noise3_accuracy), np.std(noise3_accuracy))
print(np.mean(noise4_accuracy), np.std(noise4_accuracy))

accuracy_result = np.array(
    [total_accuracy, octree_accuracy, grid_accuracy, random_accuracy, noise1_accuracy, noise2_accuracy, noise3_accuracy,
     noise4_accuracy])
#  np.save('accuracy_result_test.npy',accuracy_result)
# np.save('./xiaorong_result/block1.npy',accuracy_result)


# accuracy_result = np.load('accuracy_result_test.npy')
# total_accuracy = accuracy_result[0]
# octree_accuracy = accuracy_result[1]
# grid_accuracy =accuracy_result[2]
# random_accuracy = accuracy_result[3]
# noise1_accuracy = accuracy_result[4]
# noise2_accuracy = accuracy_result[5]
# noise3_accuracy = accuracy_result[6]
# noise4_accuracy = accuracy_result[7]
# print(total_accuracy)
# print(np.mean(total_accuracy),np.std(total_accuracy))
# print(np.mean(octree_accuracy),np.std(octree_accuracy))
# print(np.mean(grid_accuracy),np.std(grid_accuracy))
# print(np.mean(random_accuracy),np.std(random_accuracy))
# print(np.mean(noise1_accuracy),np.std(noise1_accuracy))
# print(np.mean(noise2_accuracy),np.std(noise2_accuracy))
# print(np.mean(noise3_accuracy),np.std(noise3_accuracy))
# print(np.mean(noise4_accuracy),np.std(noise4_accuracy))
