% 点云降采样   https://www.it610.com/article/1287687208714678272.htm
input_path = 'E:\\rank\\multi_level_test';
output_path='E:\\rank\\model';
fpath = fullfile(input_path, '*.ply');%读取ply格式
im_dir = dir(fpath);
im_num = length(im_dir);
%level_all = [0.15,0.25, 0.4,0.55,0.7]; data5
level_all = [0.15,0.17,0.19,0.21,0.23,0.25,0.27,0.30,0.32,0.34,0.36,0.38,0.4,0.42,0.44,0.46,0.48,0.5,0.52,0.54,0.56,0.58,0.6,0.62,0.64,0.66,0.68,0.7,0.72,0.74];   %data6
for i=1:im_num
    ptCloudIn = pcread(fullfile(input_path,im_dir(i).name)); 
    path1=fullfile(output_path,im_dir(i).name(1:end-4));
    %mkdir(path1);
    %copyfile(fullfile(input_path,im_dir(i).name),path1);
    
    edge_length = zeros(ptCloudIn.Count,1);
    location= ptCloudIn.Location;
    for j=1:ptCloudIn.Count
        [indice, ~]   = findNearestNeighbors(ptCloudIn,location(j,:),2);%寻找最近点
        edge_length(j) = norm(location(j,:) - location(indice(2),:));
    end
    average_edge_l = mean(edge_length);
    
    mkdir(fullfile(path1,'random'));
    mkdir(fullfile(path1,'gridAverage'));
    for j = 1:size(level_all,2)
        percentage =1-level_all(j);
        ptCloudOut1 = pcdownsample(ptCloudIn, 'random', percentage);
        pcwrite(ptCloudOut1,fullfile(path1,'random',['level',num2str(j),'.ply']));
        
    end
    %level_all_2=[1.2,1.4,1.65,2.0,2.5];data5
    level_all_2=[1.1,1.15,1.20,1.25,1.30,1.35,1.4,1.45,1.5,1.55,1.60,1.65,1.7,1.75,1.8,1.85,1.9,1.95,2.0,2.05,2.1,2.15,2.2,2.25,2.3,2.35,2.4,2.45,2.5,2.55];  %data6
    for j=1:size(level_all_2,2)
        gridStep=average_edge_l*level_all_2(j);
        ptCloudOut2 = pcdownsample(ptCloudIn, 'gridAverage', gridStep);
        pcwrite(ptCloudOut2,fullfile(path1,'gridAverage',['level',num2str(j),'.ply']));
    end
    %ptCloudOut3 = pcdownsample(ptCloudIn, 'nonuniformGridSample', maxNumPoints);
end
