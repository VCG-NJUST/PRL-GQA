import os
from plyfile import PlyData
import matplotlib.pyplot as plt
import numpy as np

'''
@author leon
@desc 本模块是一个测试文件，用来统计一个文件夹中点云模型的点数
@date 2021/12
'''


path = "E:\\rank\\sjtu-PCA\\distortion\\ULB Unicorn"
data = os.listdir(path)
max_number = 0
min_number = 10000
num = []
for i in data:
    plydata = PlyData.read(path+'/'+i)
    number = plydata['vertex']['x'].shape[0]
    if number > max_number :
        max_number = number
    if number < min_number :
        min_number =number
    num.append(number)
num = np.array(num)

n, bins, patches = plt.hist(x=num, bins='auto', color='#0504aa',
    alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('My Very Own Histogram')
plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# 设置y轴的上限
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()
print(max_number)
print(min_number)

# 文件重命名
# for i in range(len(data)):
#     ipath=path+'/'+data[i]
#
#     opath=path+'/'+data[i].replace(' ','_')
#     os.rename(ipath,opath)