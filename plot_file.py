import matplotlib.pyplot as plt

'''
@author leon
@desc 画图可视化
@date 2021/12
'''
lossfile_path = './loss.txt'
accfile_path ='./ave_acc.txt'
data_loss = []
data_acc = []
with open(lossfile_path, "r", encoding='utf-8') as f_txt:
    lines = f_txt.readlines()
    for line in lines:
        data_loss.append(float(line))
with open(accfile_path, "r", encoding='utf-8') as f_txt:
    lines = f_txt.readlines()
    for line in lines:
        data_acc.append(float(line))
x1=range(1,len(data_loss)+1)
x2=range(1,len(data_acc)+1)
fig1 = plt.figure("loss")
plt.plot(x1,data_loss)
fig2 =plt.figure('acc')
plt.plot(x2,data_acc)
plt.show()