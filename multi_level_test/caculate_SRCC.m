clear;
input_path = './bunny';

fpath = fullfile(input_path, '*.ply');%��ȡply��ʽ
raw_models = dir(fpath);
raw_model = pcread(fullfile(input_path,raw_models(1).name));
raw_model.Normal=pcnormals(raw_model);
dir_list = dir(input_path);
size = length(dir_list);
MOS = [1:1:30];
result.distortion_type = [];
result.srcc = [];
for i=3:size  % ǰ��������
    file = fullfile(input_path,dir_list(i).name);
    if isdir(file)
        distorton_type = dir_list(i).name;
        score=zeros(30,1);
        for j=1:30
            fpath = fullfile(file,['level',num2str(j),'.ply']);
            model = pcread(fpath);
            model.Normal=pcnormals(model);
            [asBA, asAB, asSym] =angularSimilarity(raw_model, model, 'min');
            score(j) = asSym;
        end
        if strcmp(distorton_type, 'noise3')
            linshi = score;
        end
        [srocc,krocc,plcc,rmse,or] = verify_performance(MOS,score);
        result.distortion_type = [result.distortion_type,distorton_type];
        result.srcc = [result.srcc , srocc];
    end
end


