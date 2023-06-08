import network as net
import torch.nn as nn
import torch.optim as optim
esp = 1e-8
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from fine_turn.coefficient_calu import corr_value
from data_loader import index_to_points
from scipy import io
import torch
from torch.utils.data import Dataset, DataLoader
from FPS import load_ply,farthest_point_sample,index_points,query_ball_point,relative_cordinate
import numpy as np


def pc_normalize(pc):
    '''
    :param pc:B X N x D
    :return:  零均值化 B x N x D
    '''
    centroid = np.mean(pc, axis=1)  # B X D
    B = pc.shape[0]
    data =np.zeros(pc.shape,dtype=pc.dtype)
    for batch in range(B):
        data[batch]=pc[batch]-centroid[batch]      # B x N x D
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=-1)),axis=-1)  # (B,)
    for batch in range(B):
        pc[batch] =data[batch]/m[batch]
     # 归一化到【-1 1】之间
    return pc

#处理SJTU-PCA数据集

def data_handle():
    dataFile = './sjtu-PCA/Final_MOS.mat'
    data = io.loadmat(dataFile)
    pair_txt_train = open('SJTU_train.txt', mode="w", encoding="utf-8")
    pair_txt_test = open('SJTU_test.txt', mode="w", encoding="utf-8")
    model_path='./sjtu-PCA/distortion'
    model = os.listdir(model_path)
    name_list = ['redandblack','Romanoillamp','loot','soldier','ULB_Unicorn','longdress','statue','shiva','hhi']
    dict ={}
    for i in range(len(name_list)):
        dict[name_list[i]]= i
    size = len(model)
    ratio_for_train=0.8
    for i in range(0, int(size * ratio_for_train)):   # 训练集
        dir_path=model_path+'/'+model[i]
        distortion = os.listdir(dir_path)
        for j in distortion:     # 数据读取往往不是按照顺序
            mpath = dir_path + '/' + j
            num = len(model[i])
            index = j[num+1:-4]
            index =int(index)
            type = ''
            if index >= 0 and index <= 5:
                type = 'OT'
            elif index >= 6 and index <= 11:
                type = 'CN'
                continue
            elif index >= 12 and index <= 17:
                type = 'DS'
            elif index >= 18 and index <= 23:
                type = 'D+C'
                continue
            elif index >= 24 and index <= 29:
                type = 'D+G'
            elif index >= 30 and index <= 35:
                type = 'GGN'
            else:
                type = 'C+G'
                continue
            pair_txt_train.write('%s %f %s\n' % (mpath, data['Final_MOS'][index][dict[model[i]]],type))
    for i in range(int(size * ratio_for_train), size):
        dir_path=model_path+'/'+model[i]
        distortion = os.listdir(dir_path)
        for j in distortion:     # 数据读取往往不是按照顺序
            mpath = dir_path + '/' + j
            num = len(model[i])
            index = j[num+1:-4]
            index =int(index)
            type = ''
            if index >= 0 and index <= 5:
                type = 'OT'
            elif index >= 6 and index <= 11:
                type = 'CN'
                continue
            elif index >= 12 and index <= 17:
                type = 'DS'
            elif index >= 18 and index <= 23:
                type = 'D+C'
                continue
            elif index >= 24 and index <= 29:
                type = 'D+G'
            elif index >= 30 and index <= 35:
                type = 'GGN'
            else:
                type = 'C+G'
                continue
            pair_txt_test.write('%s %f %s\n' % (mpath, data['Final_MOS'][index][dict[model[i]]],type))


def read_min_withtype(root_txt_path):   # text格式是路径+分数+失真类型
    assert root_txt_path.endswith('.txt')
    path_list, score_list ,type_list= [], [],[]
    with open(root_txt_path, "r", encoding='utf-8') as f_txt:
        lines = f_txt.readlines()  # 读取全部内容 ，并以列表方式返回
        for line in lines:
            path, score,type = line.strip().split(' ')
            score = float(score)
            path_list.append(path)
            score_list.append(score)
            type_list.append(type)
    return path_list, score_list,type_list

class SJTUDataset(Dataset):     # SJTU    catergory用于是否返回样本中的类别信息以便后期统计各类型下的性能
    def __init__(self, root_txt_path,catergory=False):
        super(SJTUDataset, self).__init__()
        assert isinstance(root_txt_path, str) and root_txt_path.endswith('.txt')
        self.category =catergory
        self.path_list, self.score_list,self.type_list = read_min_withtype(root_txt_path)

    def __getitem__(self, index):
        '''

        :param index: 样本序号
        :return:  random_sizexpatch_sizex3
        '''
        path = self.path_list[index]
        score = self.score_list[index]
        model = load_ply(path).reshape(1, -1, 3)  # ply模型    BxNx3  B=1
        model = pc_normalize(model)

        model = torch.from_numpy(model)
        centroids_index = farthest_point_sample(model, 64)  # 每次采样点数
        centroids = index_points(model, centroids_index)  # 确定采样中兴点坐标
        # radius采样
        result1 = query_ball_point(0.2, 512, model, centroids)
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
        data1_tensor = relative_cordinate(data1_tensor, centroids)
        data1_tensor = data1_tensor[0]  # S X patch_size X C
        # data_patch1=pc_normalize(data_patch1)
        # data_patch2=pc_normalize(data_patch2)  #   坐标零均值化
        label_tensor = torch.tensor(score, dtype=torch.float)
        label_tensor = label_tensor.unsqueeze(-1)
        if self.category:
            return data1_tensor, label_tensor, self.type_list[index]
        else:
            return data1_tensor, label_tensor

    def __len__(self):
        return len(self.path_list)






# 二次微调训练
def test(test_loader, net, criterion):
    loss = 0
    num = 0
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predict_mos=[]
    mos=[]
    for idx, (data1,label) in enumerate(test_loader):
        data1 = data1.to(device)
        data1 = torch.transpose(data1, -1, -2)  # dataloader中数据为Bxrandom_sizexpatch_sizex3
        label = label.to(device)
        out = net(data1)
        predict_mos.append(out.cpu().item())
        mos.append(label.cpu().item())
        loss_net = criterion(out, label)
        loss += loss_net.cpu().detach().item()*data1.size()[0]
        num+=data1.size()[0]
    mos = np.array(mos,dtype=np.float)
    predict_mos = np.array(predict_mos,dtype=np.float)
    PLCC,SRCC,KRCC,RMSE = corr_value(mos,predict_mos,False)
    print('\nTest set: Average loss: {:.4f}\n'.format(loss/num))
    print('PLCC :{0},SRCC:{1},KRCC:{2},RMSE:{3}'.format(PLCC,SRCC,KRCC,RMSE))
    return loss/num

def train_and_test():
    save_model = torch.load('../../params.pth')
    score_Net=net.weight_score()
    score_Net.load_state_dict(save_model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    score_Net=score_Net.to(device)
    train_Dataset = SJTUDataset('./SJTU_train.txt')
    test_Dataset = SJTUDataset('./SJTU_test.txt')
    trainloader = DataLoader(train_Dataset, batch_size=4, num_workers=0, shuffle=True,
                                 drop_last=False)
    testloader = DataLoader(test_Dataset, batch_size=1, num_workers=0, shuffle=False,
                                drop_last=False)
    criterion = nn.MSELoss()
    optimizer =optim.SGD(score_Net.parameters(), lr=0.001, momentum=0.9)
    epochs = 20
    best_loss = 100000
    best_epoch = 0
    lossfile_path = './MSEloss.txt'
    f_loss = open(lossfile_path, 'w', encoding='utf-8')
    for epoch in range(1, epochs + 1):
        score_Net.train()
        # loss = 0
        for idx, (data1,label) in enumerate(trainloader):
            data1 = data1.to(device)
            data1 = torch.transpose(data1, -1, -2)  # dataloader中数据为Bxrandom_sizexpatch_sizex3
            label = label.to(device)
            out = score_Net(data1)
            loss_net = criterion(out, label)
            # loss += loss_net.cpu().detach().item()
            optimizer.zero_grad()
            loss_net.backward()
            optimizer.step()
            print(loss_net.cpu().detach().item())
            # if idx % 5 == 4:
            #     print("loss:" + str(loss / 5))
            #     if f_loss is not None:
            #         f_loss.write(str(loss /5) + "\n")
            #     loss = 0
            # acc=test(train_loader,net,criterion,epoch)
        test_loss= test(testloader,score_Net,criterion)
        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            torch.save(score_Net.state_dict(), 'minNET_params.pth')
    print('\nbest_epoch: {:d}, best loss: {:.2f}%\n'.format(
            best_epoch, best_loss))
    print("finish training")
    f_loss.close()

if __name__ == '__main__':
    #data_handle()
    train_and_test()
    # path ='./SJTU_train.txt'
    # read_min_withtype(path)
