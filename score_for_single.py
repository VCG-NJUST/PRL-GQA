import torch
import network as net
import torch.nn as nn
import os
esp = 1e-8
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from FPS import load_ply, farthest_point_sample, index_points, query_ball_point, relative_cordinate
import numpy as np
from data_loader import index_to_points
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

'''
@author leon
@desc 将测试集的数据用单个分支计算预测的分数值，数据集已经从pair转为single
@date 2022/1
'''


def read_min_txt(root_txt_path):  # text格式是路径+分数
    assert root_txt_path.endswith('.txt')
    path_list = []
    with open(root_txt_path, "r", encoding='utf-8') as f_txt:
        lines = f_txt.readlines()  # 读取全部内容 ，并以列表方式返回
        for line in lines:
            path = line.strip()
            path_list.append(path)
    return path_list


class SingleDataset(Dataset):  # GPCD
    def __init__(self, root_txt_path):
        super(SingleDataset, self).__init__()
        assert isinstance(root_txt_path, str) and root_txt_path.endswith('.txt')
        self.path_list = read_min_txt(root_txt_path)

    def __getitem__(self, index):
        '''

        :param index: 样本序号
        :return:  random_sizexpatch_sizex3
        '''
        path = self.path_list[index]
        model = load_ply(path).reshape(1, -1, 3)  # ply模型    BxNx3  B=1
        model = torch.from_numpy(model)
        centroids_index = farthest_point_sample(model, 64)  # 每次采样点数
        centroids = index_points(model, centroids_index)  # 确定采样中兴点坐标
        # radius采样
        result1 = query_ball_point(0.2, 516, model, centroids)
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
        data1_tensor = relative_cordinate(data1_tensor, centroids)
        data1_tensor = data1_tensor[0]  # S X patch_size X C
        # data_patch1=pc_normalize(data_patch1)
        # data_patch2=pc_normalize(data_patch2)  #   坐标零均值化
        return data1_tensor

    def __len__(self):
        return len(self.path_list)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# save_model = torch.load('params.pth')
save_model = torch.load('new_params.pth')
score_Net = net.weight_score()
score_Net.load_state_dict(save_model)
score_Net.to(device)
# test_Dataset = SingleDataset('./single_test.txt')
test_Dataset = SingleDataset('./single_test_gpcd.txt')
test_loader = DataLoader(test_Dataset, batch_size=1, num_workers=0, shuffle=False,
                         drop_last=False)
score_Net.eval()
# scores = open('./scores_for_test.txt', mode="w", encoding="utf-8")


if not os.path.exists('tmp'):
    os.mkdir('tmp')
for i in range(50):
    pbar = tqdm(test_loader)
    scores = open('./tmp/scores_for_test_gpcd_%d.txt' % i, mode="w", encoding="utf-8")
    for idx, data1 in enumerate(pbar):
        pbar.set_description('Processing %2d/50:' % i)
        data1 = data1.to(device)
        data1 = torch.transpose(data1, -1, -2)  # dataloader中数据为Bxrandom_sizexpatch_sizex3
        out = score_Net(data1)
        scores.write('%s\n' % (str(out.cpu().item())))
