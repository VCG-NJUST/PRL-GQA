clear;
input_txt =importdata('../rank_pair_test.txt');
size = length(input_txt);

accuracy_octree = [];
accuracy_grid=[];
accuracy_random = [];
accuracy_noise1 = [];
accuracy_noise2 = [];
accuracy_noise3 = [];
accuracy_noise4 = [];
accuracy_all = [];


correct_number_octree=0;
correct_number_grid=0;
correct_number_random = 0;
correct_number_noise1 = 0;
correct_number_noise2 = 0;
correct_number_noise3 = 0;
correct_number_noise4 = 0; 
number_octree=0;
number_grid=0;
number_random = 0;
number_noise1 = 0;
number_noise2 = 0;
number_noise3 = 0;
number_noise4 = 0; 
for i = 1:size
    str = input_txt{i};
    str = strsplit(str,' ');
    raw_model =pcread(['.',str{1}]);
    model1 =pcread(['.',str{2}]);
    model2 =pcread(['.',str{3}]);
    type = str{5};
    if strcmp(type, 'OctreeCom')
        number_octree =number_octree+1;
    elseif strcmp(type, 'gridAverage')
        number_grid =number_grid+1;
    elseif strcmp(type, 'random')
        number_random=number_random+1;
    elseif strcmp(type, 'noise1')
        number_noise1 =number_noise1+1;
    elseif strcmp(type, 'noise2')
        number_noise2 =number_noise2+1;
    elseif strcmp(type, 'noise3')
        number_noise3 =number_noise3+1;    
    elseif strcmp(type, 'noise4')
        number_noise4 =number_noise4+1; 
    end
    
    raw_model.Normal=pcnormals(raw_model);
    model1.Normal=pcnormals(model1);
    model2.Normal=pcnormals(model2);
    [asBA1, asAB1, asSym1] = point_to_point(raw_model, model1, 'mse');
    [asBA2, asAB2, asSym2] = point_to_point(raw_model, model2, 'mse');
    
    if asSym1 < asSym2
         if strcmp(type, 'OctreeCom')
               correct_number_octree =correct_number_octree+1;
         elseif strcmp(type, 'gridAverage')
               correct_number_grid =correct_number_grid+1;
         elseif strcmp(type, 'random')
               correct_number_random=correct_number_random+1;
         elseif strcmp(type, 'noise1')
               correct_number_noise1 =correct_number_noise1+1;
         elseif strcmp(type, 'noise2')
               correct_number_noise2 =correct_number_noise2+1;
         elseif strcmp(type, 'noise3')
              correct_number_noise3 =correct_number_noise3+1;    
         elseif strcmp(type, 'noise4')
              correct_number_noise4 =correct_number_noise4+1; 
         end
    end
end

accuracy_octree = [accuracy_octree,correct_number_octree/number_octree];
accuracy_grid=[accuracy_grid,correct_number_grid/number_grid];
accuracy_random = [accuracy_random,correct_number_random/number_random];
accuracy_noise1 = [accuracy_noise1,correct_number_noise1/number_noise1];
accuracy_noise2 = [accuracy_noise2,correct_number_noise2/number_noise2];
accuracy_noise3 = [accuracy_noise3,correct_number_noise3/number_noise3];
accuracy_noise4 = [accuracy_noise4,correct_number_noise4/number_noise4];
accuracy_all = [accuracy_all,(correct_number_octree+correct_number_grid+correct_number_random+correct_number_noise1+correct_number_noise2+correct_number_noise3+correct_number_noise4)/size];



correct_number_octree=0;
correct_number_grid=0;
correct_number_random = 0;
correct_number_noise1 = 0;
correct_number_noise2 = 0;
correct_number_noise3 = 0;
correct_number_noise4 = 0; 
number_octree=0;
number_grid=0;
number_random = 0;
number_noise1 = 0;
number_noise2 = 0;
number_noise3 = 0;
number_noise4 = 0;  
for i = 1:size
    str = input_txt{i};
    str = strsplit(str,' ');
    raw_model =pcread(['.',str{1}]);
    model1 =pcread(['.',str{2}]);
    model2 =pcread(['.',str{3}]);
    type = str{5};
    if strcmp(type, 'OctreeCom')
        number_octree =number_octree+1;
    elseif strcmp(type, 'gridAverage')
        number_grid =number_grid+1;
    elseif strcmp(type, 'random')
        number_random=number_random+1;
    elseif strcmp(type, 'noise1')
        number_noise1 =number_noise1+1;
    elseif strcmp(type, 'noise2')
        number_noise2 =number_noise2+1;
    elseif strcmp(type, 'noise3')
        number_noise3 =number_noise3+1;    
    elseif strcmp(type, 'noise4')
        number_noise4 =number_noise4+1; 
    end
    raw_model.Normal=pcnormals(raw_model);
    model1.Normal=pcnormals(model1);
    model2.Normal=pcnormals(model2);
    [asBA1, asAB1, asSym1] = point_to_point(raw_model, model1, 'max');
    [asBA2, asAB2, asSym2] = point_to_point(raw_model, model2, 'max');
    if asSym1 < asSym2
         if strcmp(type, 'OctreeCom')
               correct_number_octree =correct_number_octree+1;
         elseif strcmp(type, 'gridAverage')
               correct_number_grid =correct_number_grid+1;
         elseif strcmp(type, 'random')
               correct_number_random=correct_number_random+1;
         elseif strcmp(type, 'noise1')
               correct_number_noise1 =correct_number_noise1+1;
         elseif strcmp(type, 'noise2')
               correct_number_noise2 =correct_number_noise2+1;
         elseif strcmp(type, 'noise3')
              correct_number_noise3 =correct_number_noise3+1;    
         elseif strcmp(type, 'noise4')
              correct_number_noise4 =correct_number_noise4+1; 
         end
    end
end
accuracy_octree = [accuracy_octree,correct_number_octree/number_octree];
accuracy_grid=[accuracy_grid,correct_number_grid/number_grid];
accuracy_random = [accuracy_random,correct_number_random/number_random];
accuracy_noise1 = [accuracy_noise1,correct_number_noise1/number_noise1];
accuracy_noise2 = [accuracy_noise2,correct_number_noise2/number_noise2];
accuracy_noise3 = [accuracy_noise3,correct_number_noise3/number_noise3];
accuracy_noise4 = [accuracy_noise4,correct_number_noise4/number_noise4];
accuracy_all = [accuracy_all,(correct_number_octree+correct_number_grid+correct_number_random+correct_number_noise1+correct_number_noise2+correct_number_noise3+correct_number_noise4)/size];



correct_number_octree=0;
correct_number_grid=0;
correct_number_random = 0;
correct_number_noise1 = 0;
correct_number_noise2 = 0;
correct_number_noise3 = 0;
correct_number_noise4 = 0; 
number_octree=0;
number_grid=0;
number_random = 0;
number_noise1 = 0;
number_noise2 = 0;
number_noise3 = 0;
number_noise4 = 0; 
for i = 1:size
    str = input_txt{i};
    str = strsplit(str,' ');
    raw_model =pcread(['.',str{1}]);
    model1 =pcread(['.',str{2}]);
    model2 =pcread(['.',str{3}]);
    type = str{5};
    if strcmp(type, 'OctreeCom')
        number_octree =number_octree+1;
    elseif strcmp(type, 'gridAverage')
        number_grid =number_grid+1;
    elseif strcmp(type, 'random')
        number_random=number_random+1;
    elseif strcmp(type, 'noise1')
        number_noise1 =number_noise1+1;
    elseif strcmp(type, 'noise2')
        number_noise2 =number_noise2+1;
    elseif strcmp(type, 'noise3')
        number_noise3 =number_noise3+1;    
    elseif strcmp(type, 'noise4')
        number_noise4 =number_noise4+1; 
    end
    raw_model.Normal=pcnormals(raw_model);
    model1.Normal=pcnormals(model1);
    model2.Normal=pcnormals(model2);
    [asBA1, asAB1, asSym1] = point_to_point(raw_model, model1, 'PSNR_MSE');
    [asBA2, asAB2, asSym2] = point_to_point(raw_model, model2, 'PSNR_MSE');
    if asSym1 < asSym2
         if strcmp(type, 'OctreeCom')
               correct_number_octree =correct_number_octree+1;
         elseif strcmp(type, 'gridAverage')
               correct_number_grid =correct_number_grid+1;
         elseif strcmp(type, 'random')
               correct_number_random=correct_number_random+1;
         elseif strcmp(type, 'noise1')
               correct_number_noise1 =correct_number_noise1+1;
         elseif strcmp(type, 'noise2')
               correct_number_noise2 =correct_number_noise2+1;
         elseif strcmp(type, 'noise3')
              correct_number_noise3 =correct_number_noise3+1;    
         elseif strcmp(type, 'noise4')
              correct_number_noise4 =correct_number_noise4+1; 
         end
    end
end
accuracy_octree = [accuracy_octree,correct_number_octree/number_octree];
accuracy_grid=[accuracy_grid,correct_number_grid/number_grid];
accuracy_random = [accuracy_random,correct_number_random/number_random];
accuracy_noise1 = [accuracy_noise1,correct_number_noise1/number_noise1];
accuracy_noise2 = [accuracy_noise2,correct_number_noise2/number_noise2];
accuracy_noise3 = [accuracy_noise3,correct_number_noise3/number_noise3];
accuracy_noise4 = [accuracy_noise4,correct_number_noise4/number_noise4];
accuracy_all = [accuracy_all,(correct_number_octree+correct_number_grid+correct_number_random+correct_number_noise1+correct_number_noise2+correct_number_noise3+correct_number_noise4)/size];




correct_number_octree=0;
correct_number_grid=0;
correct_number_random = 0;
correct_number_noise1 = 0;
correct_number_noise2 = 0;
correct_number_noise3 = 0;
correct_number_noise4 = 0; 
number_octree=0;
number_grid=0;
number_random = 0;
number_noise1 = 0;
number_noise2 = 0;
number_noise3 = 0;
number_noise4 = 0;  
for i = 1:size
    str = input_txt{i};
    str = strsplit(str,' ');
    raw_model =pcread(['.',str{1}]);
    model1 =pcread(['.',str{2}]);
    model2 =pcread(['.',str{3}]);
    type = str{5};
    if strcmp(type, 'OctreeCom')
        number_octree =number_octree+1;
    elseif strcmp(type, 'gridAverage')
        number_grid =number_grid+1;
    elseif strcmp(type, 'random')
        number_random=number_random+1;
    elseif strcmp(type, 'noise1')
        number_noise1 =number_noise1+1;
    elseif strcmp(type, 'noise2')
        number_noise2 =number_noise2+1;
    elseif strcmp(type, 'noise3')
        number_noise3 =number_noise3+1;    
    elseif strcmp(type, 'noise4')
        number_noise4 =number_noise4+1; 
    end
    raw_model.Normal=pcnormals(raw_model);
    model1.Normal=pcnormals(model1);
    model2.Normal=pcnormals(model2);
    [asBA1, asAB1, asSym1] = point_to_plane(raw_model, model1, 'mse');
    [asBA2, asAB2, asSym2] = point_to_plane(raw_model, model2, 'mse');
    if asSym1 < asSym2
         if strcmp(type, 'OctreeCom')
               correct_number_octree =correct_number_octree+1;
         elseif strcmp(type, 'gridAverage')
               correct_number_grid =correct_number_grid+1;
         elseif strcmp(type, 'random')
               correct_number_random=correct_number_random+1;
         elseif strcmp(type, 'noise1')
               correct_number_noise1 =correct_number_noise1+1;
         elseif strcmp(type, 'noise2')
               correct_number_noise2 =correct_number_noise2+1;
         elseif strcmp(type, 'noise3')
              correct_number_noise3 =correct_number_noise3+1;    
         elseif strcmp(type, 'noise4')
              correct_number_noise4 =correct_number_noise4+1; 
         end
    end
end
accuracy_octree = [accuracy_octree,correct_number_octree/number_octree];
accuracy_grid=[accuracy_grid,correct_number_grid/number_grid];
accuracy_random = [accuracy_random,correct_number_random/number_random];
accuracy_noise1 = [accuracy_noise1,correct_number_noise1/number_noise1];
accuracy_noise2 = [accuracy_noise2,correct_number_noise2/number_noise2];
accuracy_noise3 = [accuracy_noise3,correct_number_noise3/number_noise3];
accuracy_noise4 = [accuracy_noise4,correct_number_noise4/number_noise4];
accuracy_all = [accuracy_all,(correct_number_octree+correct_number_grid+correct_number_random+correct_number_noise1+correct_number_noise2+correct_number_noise3+correct_number_noise4)/size];

correct_number_octree=0;
correct_number_grid=0;
correct_number_random = 0;
correct_number_noise1 = 0;
correct_number_noise2 = 0;
correct_number_noise3 = 0;
correct_number_noise4 = 0; 
number_octree=0;
number_grid=0;
number_random = 0;
number_noise1 = 0;
number_noise2 = 0;
number_noise3 = 0;
number_noise4 = 0;  
for i = 1:size
    str = input_txt{i};
    str = strsplit(str,' ');
    raw_model =pcread(['.',str{1}]);
    model1 =pcread(['.',str{2}]);
    model2 =pcread(['.',str{3}]);
    type = str{5};
    if strcmp(type, 'OctreeCom')
        number_octree =number_octree+1;
    elseif strcmp(type, 'gridAverage')
        number_grid =number_grid+1;
    elseif strcmp(type, 'random')
        number_random=number_random+1;
    elseif strcmp(type, 'noise1')
        number_noise1 =number_noise1+1;
    elseif strcmp(type, 'noise2')
        number_noise2 =number_noise2+1;
    elseif strcmp(type, 'noise3')
        number_noise3 =number_noise3+1;    
    elseif strcmp(type, 'noise4')
        number_noise4 =number_noise4+1; 
    end
    raw_model.Normal=pcnormals(raw_model);
    model1.Normal=pcnormals(model1);
    model2.Normal=pcnormals(model2);
    [asBA1, asAB1, asSym1] = point_to_plane(raw_model, model1, 'max');
    [asBA2, asAB2, asSym2] = point_to_plane(raw_model, model2, 'max');
    if asSym1 < asSym2
          if strcmp(type, 'OctreeCom')
               correct_number_octree =correct_number_octree+1;
         elseif strcmp(type, 'gridAverage')
               correct_number_grid =correct_number_grid+1;
         elseif strcmp(type, 'random')
               correct_number_random=correct_number_random+1;
         elseif strcmp(type, 'noise1')
               correct_number_noise1 =correct_number_noise1+1;
         elseif strcmp(type, 'noise2')
               correct_number_noise2 =correct_number_noise2+1;
         elseif strcmp(type, 'noise3')
              correct_number_noise3 =correct_number_noise3+1;    
         elseif strcmp(type, 'noise4')
              correct_number_noise4 =correct_number_noise4+1; 
         end
    end
end
accuracy_octree = [accuracy_octree,correct_number_octree/number_octree];
accuracy_grid=[accuracy_grid,correct_number_grid/number_grid];
accuracy_random = [accuracy_random,correct_number_random/number_random];
accuracy_noise1 = [accuracy_noise1,correct_number_noise1/number_noise1];
accuracy_noise2 = [accuracy_noise2,correct_number_noise2/number_noise2];
accuracy_noise3 = [accuracy_noise3,correct_number_noise3/number_noise3];
accuracy_noise4 = [accuracy_noise4,correct_number_noise4/number_noise4];
accuracy_all = [accuracy_all,(correct_number_octree+correct_number_grid+correct_number_random+correct_number_noise1+correct_number_noise2+correct_number_noise3+correct_number_noise4)/size];

correct_number_octree=0;
correct_number_grid=0;
correct_number_random = 0;
correct_number_noise1 = 0;
correct_number_noise2 = 0;
correct_number_noise3 = 0;
correct_number_noise4 = 0; 
number_octree=0;
number_grid=0;
number_random = 0;
number_noise1 = 0;
number_noise2 = 0;
number_noise3 = 0;
number_noise4 = 0; 
for i = 1:size
    str = input_txt{i};
    str = strsplit(str,' ');
    raw_model =pcread(['.',str{1}]);
    model1 =pcread(['.',str{2}]);
    model2 =pcread(['.',str{3}]);
    type = str{5};
    if strcmp(type, 'OctreeCom')
        number_octree =number_octree+1;
    elseif strcmp(type, 'gridAverage')
        number_grid =number_grid+1;
    elseif strcmp(type, 'random')
        number_random=number_random+1;
    elseif strcmp(type, 'noise1')
        number_noise1 =number_noise1+1;
    elseif strcmp(type, 'noise2')
        number_noise2 =number_noise2+1;
    elseif strcmp(type, 'noise3')
        number_noise3 =number_noise3+1;    
    elseif strcmp(type, 'noise4')
        number_noise4 =number_noise4+1; 
    end
    raw_model.Normal=pcnormals(raw_model);
    model1.Normal=pcnormals(model1);
    model2.Normal=pcnormals(model2);
    [asBA1, asAB1, asSym1] = point_to_plane(raw_model, model1, 'PSNR_MSE');
    [asBA2, asAB2, asSym2] = point_to_plane(raw_model, model2, 'PSNR_MSE');
    if asSym1 < asSym2
         if strcmp(type, 'OctreeCom')
               correct_number_octree =correct_number_octree+1;
         elseif strcmp(type, 'gridAverage')
               correct_number_grid =correct_number_grid+1;
         elseif strcmp(type, 'random')
               correct_number_random=correct_number_random+1;
         elseif strcmp(type, 'noise1')
               correct_number_noise1 =correct_number_noise1+1;
         elseif strcmp(type, 'noise2')
               correct_number_noise2 =correct_number_noise2+1;
         elseif strcmp(type, 'noise3')
              correct_number_noise3 =correct_number_noise3+1;    
         elseif strcmp(type, 'noise4')
              correct_number_noise4 =correct_number_noise4+1; 
         end
    end
end
accuracy_octree = [accuracy_octree,correct_number_octree/number_octree];
accuracy_grid=[accuracy_grid,correct_number_grid/number_grid];
accuracy_random = [accuracy_random,correct_number_random/number_random];
accuracy_noise1 = [accuracy_noise1,correct_number_noise1/number_noise1];
accuracy_noise2 = [accuracy_noise2,correct_number_noise2/number_noise2];
accuracy_noise3 = [accuracy_noise3,correct_number_noise3/number_noise3];
accuracy_noise4 = [accuracy_noise4,correct_number_noise4/number_noise4];
accuracy_all = [accuracy_all,(correct_number_octree+correct_number_grid+correct_number_random+correct_number_noise1+correct_number_noise2+correct_number_noise3+correct_number_noise4)/size];

correct_number_octree=0;
correct_number_grid=0;
correct_number_random = 0;
correct_number_noise1 = 0;
correct_number_noise2 = 0;
correct_number_noise3 = 0;
correct_number_noise4 = 0; 
number_octree=0;
number_grid=0;
number_random = 0;
number_noise1 = 0;
number_noise2 = 0;
number_noise3 = 0;
number_noise4 = 0; 
for i = 1:size
    str = input_txt{i};
    str = strsplit(str,' ');
    raw_model =pcread(['.',str{1}]);
    model1 =pcread(['.',str{2}]);
    model2 =pcread(['.',str{3}]);
    type = str{5};
    if strcmp(type, 'OctreeCom')
        number_octree =number_octree+1;
    elseif strcmp(type, 'gridAverage')
        number_grid =number_grid+1;
    elseif strcmp(type, 'random')
        number_random=number_random+1;
    elseif strcmp(type, 'noise1')
        number_noise1 =number_noise1+1;
    elseif strcmp(type, 'noise2')
        number_noise2 =number_noise2+1;
    elseif strcmp(type, 'noise3')
        number_noise3 =number_noise3+1;    
    elseif strcmp(type, 'noise4')
        number_noise4 =number_noise4+1; 
    end
    raw_model.Normal=pcnormals(raw_model);
    model1.Normal=pcnormals(model1);
    model2.Normal=pcnormals(model2);
    [asBA1, asAB1, asSym1] = angularSimilarity(raw_model, model1, 'mse');
    [asBA2, asAB2, asSym2] = angularSimilarity(raw_model, model2, 'mse');
    if asSym1 < asSym2
         if strcmp(type, 'OctreeCom')
               correct_number_octree =correct_number_octree+1;
         elseif strcmp(type, 'gridAverage')
               correct_number_grid =correct_number_grid+1;
         elseif strcmp(type, 'random')
               correct_number_random=correct_number_random+1;
         elseif strcmp(type, 'noise1')
               correct_number_noise1 =correct_number_noise1+1;
         elseif strcmp(type, 'noise2')
               correct_number_noise2 =correct_number_noise2+1;
         elseif strcmp(type, 'noise3')
              correct_number_noise3 =correct_number_noise3+1;    
         elseif strcmp(type, 'noise4')
              correct_number_noise4 =correct_number_noise4+1; 
         end
    end
end
accuracy_octree = [accuracy_octree,correct_number_octree/number_octree];
accuracy_grid=[accuracy_grid,correct_number_grid/number_grid];
accuracy_random = [accuracy_random,correct_number_random/number_random];
accuracy_noise1 = [accuracy_noise1,correct_number_noise1/number_noise1];
accuracy_noise2 = [accuracy_noise2,correct_number_noise2/number_noise2];
accuracy_noise3 = [accuracy_noise3,correct_number_noise3/number_noise3];
accuracy_noise4 = [accuracy_noise4,correct_number_noise4/number_noise4];
accuracy_all = [accuracy_all,(correct_number_octree+correct_number_grid+correct_number_random+correct_number_noise1+correct_number_noise2+correct_number_noise3+correct_number_noise4)/size];



correct_number_octree=0;
correct_number_grid=0;
correct_number_random = 0;
correct_number_noise1 = 0;
correct_number_noise2 = 0;
correct_number_noise3 = 0;
correct_number_noise4 = 0; 
number_octree=0;
number_grid=0;
number_random = 0;
number_noise1 = 0;
number_noise2 = 0;
number_noise3 = 0;
number_noise4 = 0; 
for i = 1:size
    str = input_txt{i};
    str = strsplit(str,' ');
    raw_model =pcread(['.',str{1}]);
    model1 =pcread(['.',str{2}]);
    model2 =pcread(['.',str{3}]);
    type = str{5};
    if strcmp(type, 'OctreeCom')
        number_octree =number_octree+1;
    elseif strcmp(type, 'gridAverage')
        number_grid =number_grid+1;
    elseif strcmp(type, 'random')
        number_random=number_random+1;
    elseif strcmp(type, 'noise1')
        number_noise1 =number_noise1+1;
    elseif strcmp(type, 'noise2')
        number_noise2 =number_noise2+1;
    elseif strcmp(type, 'noise3')
        number_noise3 =number_noise3+1;    
    elseif strcmp(type, 'noise4')
        number_noise4 =number_noise4+1; 
    end
    raw_model.Normal=pcnormals(raw_model);
    model1.Normal=pcnormals(model1);
    model2.Normal=pcnormals(model2);
    [asBA1, asAB1, asSym1] =angularSimilarity(raw_model, model1, 'min');
    [asBA2, asAB2, asSym2] =angularSimilarity(raw_model, model2, 'min');
    if asSym1 < asSym2
         if strcmp(type, 'OctreeCom')
               correct_number_octree =correct_number_octree+1;
         elseif strcmp(type, 'gridAverage')
               correct_number_grid =correct_number_grid+1;
         elseif strcmp(type, 'random')
               correct_number_random=correct_number_random+1;
         elseif strcmp(type, 'noise1')
               correct_number_noise1 =correct_number_noise1+1;
         elseif strcmp(type, 'noise2')
               correct_number_noise2 =correct_number_noise2+1;
         elseif strcmp(type, 'noise3')
              correct_number_noise3 =correct_number_noise3+1;    
         elseif strcmp(type, 'noise4')
              correct_number_noise4 =correct_number_noise4+1; 
         end
    end
end
accuracy_octree = [accuracy_octree,correct_number_octree/number_octree];
accuracy_grid=[accuracy_grid,correct_number_grid/number_grid];
accuracy_random = [accuracy_random,correct_number_random/number_random];
accuracy_noise1 = [accuracy_noise1,correct_number_noise1/number_noise1];
accuracy_noise2 = [accuracy_noise2,correct_number_noise2/number_noise2];
accuracy_noise3 = [accuracy_noise3,correct_number_noise3/number_noise3];
accuracy_noise4 = [accuracy_noise4,correct_number_noise4/number_noise4];
accuracy_all = [accuracy_all,(correct_number_octree+correct_number_grid+correct_number_random+correct_number_noise1+correct_number_noise2+correct_number_noise3+correct_number_noise4)/size];

correct_number_octree=0;
correct_number_grid=0;
correct_number_random = 0;
correct_number_noise1 = 0;
correct_number_noise2 = 0;
correct_number_noise3 = 0;
correct_number_noise4 = 0; 
number_octree=0;
number_grid=0;
number_random = 0;
number_noise1 = 0;
number_noise2 = 0;
number_noise3 = 0;
number_noise4 = 0; 
for i = 1:size
    str = input_txt{i};
    str = strsplit(str,' ');
    raw_model =pcread(['.',str{1}]);
    model1 =pcread(['.',str{2}]);
    model2 =pcread(['.',str{3}]);
    type = str{5};
    if strcmp(type, 'OctreeCom')
        number_octree =number_octree+1;
    elseif strcmp(type, 'gridAverage')
        number_grid =number_grid+1;
    elseif strcmp(type, 'random')
        number_random=number_random+1;
    elseif strcmp(type, 'noise1')
        number_noise1 =number_noise1+1;
    elseif strcmp(type, 'noise2')
        number_noise2 =number_noise2+1;
    elseif strcmp(type, 'noise3')
        number_noise3 =number_noise3+1;    
    elseif strcmp(type, 'noise4')
        number_noise4 =number_noise4+1; 
    end
    raw_model.Normal=pcnormals(raw_model);
    model1.Normal=pcnormals(model1);
    model2.Normal=pcnormals(model2);
    [asBA1, asAB1, asSym1] = angularSimilarity(raw_model, model1, 'PSNR_MSE');
    [asBA2, asAB2, asSym2] = angularSimilarity(raw_model, model2, 'PSNR_MSE');
    if asSym1 < asSym2
         if strcmp(type, 'OctreeCom')
               correct_number_octree =correct_number_octree+1;
         elseif strcmp(type, 'gridAverage')
               correct_number_grid =correct_number_grid+1;
         elseif strcmp(type, 'random')
               correct_number_random=correct_number_random+1;
         elseif strcmp(type, 'noise1')
               correct_number_noise1 =correct_number_noise1+1;
         elseif strcmp(type, 'noise2')
               correct_number_noise2 =correct_number_noise2+1;
         elseif strcmp(type, 'noise3')
              correct_number_noise3 =correct_number_noise3+1;    
         elseif strcmp(type, 'noise4')
              correct_number_noise4 =correct_number_noise4+1; 
         end
    end
end
accuracy_octree = [accuracy_octree,correct_number_octree/number_octree];
accuracy_grid=[accuracy_grid,correct_number_grid/number_grid];
accuracy_random = [accuracy_random,correct_number_random/number_random];
accuracy_noise1 = [accuracy_noise1,correct_number_noise1/number_noise1];
accuracy_noise2 = [accuracy_noise2,correct_number_noise2/number_noise2];
accuracy_noise3 = [accuracy_noise3,correct_number_noise3/number_noise3];
accuracy_noise4 = [accuracy_noise4,correct_number_noise4/number_noise4];
accuracy_all = [accuracy_all,(correct_number_octree+correct_number_grid+correct_number_random+correct_number_noise1+correct_number_noise2+correct_number_noise3+correct_number_noise4)/size];
