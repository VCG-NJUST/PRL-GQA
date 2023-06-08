clear;
input_path = 'E:\\rank\\multi_level_test';
output_path='E:\\rank\\model';
fpath = fullfile(input_path, '*.ply');%读取ply格式
im_dir = dir(fpath);
im_num = length(im_dir);
%noise_level_all = [0.15,0.3, 0.5,0.75,1];  data 第一次生成数据集参数
%noise_level_all = [0.10,0.20, 0.35,0.5,0.7];  data5参数
noise_level_all = [0.10,0.12,0.14,0.16,0.18,0.20,0.22,0.25,0.27,0.29,0.31,0.33,0.35,0.37,0.40,0.43,0.48,0.50,0.52,0.55,0.58,0.60,0.62,0.64,0.66,0.68,0.69,0.71,0.73,0.75];
for i=1:im_num
    orignal = pcread(fullfile(input_path,im_dir(i).name));
    location= orignal.Location;
    path1=fullfile(output_path,im_dir(i).name(1:end-4));
    mkdir(path1);
	copyfile(fullfile(input_path,im_dir(i).name),path1);
    mkdir(fullfile(path1,'noise1'));
    mkdir(fullfile(path1,'noise2'));
    mkdir(fullfile(path1,'noise3'));
    mkdir(fullfile(path1,'noise4'));
    % Sigma = [0.5 0 0;0 0.5 0;0 0 0.5]; R = chol(Sigma);%噪声的协方差
    % delta_p = randn(100,3)*R;%噪声矩阵
    
    
    edge_length = zeros(orignal.Count,1);
    for j=1:orignal.Count
        [indice, ~]   = findNearestNeighbors(orignal,location(j,:),2);%寻找最近点
        edge_length(j) = norm(location(j,:) - location(indice(2),:));
    end
    average_edge_l = mean(edge_length);
    
    %添加多个等级噪声保存成不同模型
    for j = 1:size(noise_level_all,2)
        noise_level = noise_level_all(j);
        s_noise = normrnd(0,noise_level*average_edge_l,size(location,1),3);%加噪音，其中mean parameter mu = 0, sigma就是标准正态分布的幅值 sigma = 0.01
        pc_noise = bsxfun(@plus,location,s_noise);
        ptCloud = pointCloud(pc_noise);
        pcwrite(ptCloud,fullfile(path1,'noise1',['level',num2str(j),'.ply']));
        %均匀噪声
        b=average_edge_l*noise_level*3;
        a=-average_edge_l*noise_level*3;
        s_noise =a+(b-a)*rand(size(location,1),3);
        pc_noise = bsxfun(@plus,location,s_noise);
        ptCloud = pointCloud(pc_noise);
        pcwrite(ptCloud,fullfile(path1,'noise2',['level',num2str(j),'.ply']));
        %脉冲噪声
        b=average_edge_l*noise_level*3;
        a=-average_edge_l*noise_level*3;
        p=0.2;%噪声密度
        x =rand(size(location,1),3);
        f =zeros(size(location,1),3);
        f(x<p/2)=a;
        f((x>p/2)&(x<p))=b;
        pc_noise = bsxfun(@plus,location,f);
        ptCloud = pointCloud(pc_noise);
        pcwrite(ptCloud,fullfile(path1,'noise3',['level',num2str(j),'.ply']));
        %指数噪声
        a=average_edge_l*noise_level;
        s_noise=exprnd(a,size(location,1),3);
        pc_noise = bsxfun(@plus,location,s_noise);
        ptCloud = pointCloud(pc_noise);
        pcwrite(ptCloud,fullfile(path1,'noise4',['level',num2str(j),'.ply']));
        %伽马噪声
        %noise_level=noise_level_all*3;
        %b=average_edge_l*noise_level;
        %a=average_edge_l*average_edge_l+1;
        %s_noise=gamrnd(a,1,size(location,1),3);
    end

end

    
    