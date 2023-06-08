%clear;
stimuli_path='E:\点云\数据集\G-PCD\stimuli\';
MOS=mean(subjdesktopdsis,2);
model_data=dir(fullfile(stimuli_path,'*.ply'));
model_number=length(model_data);
t=0;
result=zeros(40,1);
for i=1:model_number
    if mod(i,9)~=1
        t=t+1;
        A=pcread([stimuli_path model_data(floor((i-1)/9)*9+1).name]);
        B=pcread([stimuli_path model_data(i).name]);
        A.Normal=pcnormals(A);
        B.Normal=pcnormals(B);
        [asBA, asAB, asSym] = angularSimilarity(A, B, 'mse');
        result(t)=asSym;
    end 
end
a=[1:1:4,9:1:12,17:1:20,25:1:28,33:1:36];
b=[5:1:8,13:1:16,21:1:24,29:1:32,37:1:40];
c=[1:1:8];
d=[9:1:16];
e=[17:1:24];
f=[25:1:32];
g=[33:1:40];


[srocc3,krocc3,plcc3,rmse3,or3] = verify_performance(MOS,result);
[srocc2,krocc2,plcc2,rmse2,or2] = verify_performance(MOS(a),result(a));  %计算八叉树压缩的系数
[srocc1,krocc1,plcc1,rmse1,or1] = verify_performance(MOS(b),result(b));  %计算高斯噪声样本的系数
[srocc4,krocc4,plcc4,rmse4,or4] = verify_performance(MOS(c),result(c));  
[srocc5,krocc5,plcc5,rmse5,or5] = verify_performance(MOS(d),result(d));  
[srocc6,krocc6,plcc6,rmse6,or6] = verify_performance(MOS(e),result(e));
[srocc7,krocc7,plcc7,rmse7,or7] = verify_performance(MOS(f),result(f)); 
[srocc8,krocc8,plcc8,rmse8,or8] = verify_performance(MOS(g),result(g));   