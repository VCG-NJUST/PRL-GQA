clear;
input_txt =importdata('../rank_pair_test.txt');
size = length(input_txt);
correct_number_com_dow = 0;
correct_number_noise = 0;
com_dow_size = 0;
noise_size = 0;  
for i = 1:size
    str = input_txt{i};
    str = strsplit(str,' ');
    raw_model =pcread(['.',str{1}]);
    model1 =pcread(['.',str{2}]);
    model2 =pcread(['.',str{3}]);
    type = str{5};
    if strcmp(type, 'com&down')
        com_dow_size =com_dow_size+1;
    else
        noise_size =noise_size + 1;
    end
    raw_model.Normal=pcnormals(raw_model);
    model1.Normal=pcnormals(model1);
    model2.Normal=pcnormals(model2);
    [asBA1, asAB1, asSym1] = point_to_point(raw_model, model1, 'mse');
    [asBA2, asAB2, asSym2] = point_to_point(raw_model, model2, 'mse');
    if asSym1 < asSym2
        if strcmp(type, 'com&down')
            correct_number_com_dow=correct_number_com_dow+1;
        else
            correct_number_noise=correct_number_noise+1;
        end
    end
end
accuracy1_po2po_mse_com_dow = correct_number_com_dow/com_dow_size;
accuracy2_po2po_mse_noise = correct_number_noise/noise_size;
accuracy_po2po_mse = (correct_number_com_dow+correct_number_noise)/size;

correct_number_com_dow = 0;
correct_number_noise = 0;
com_dow_size = 0;
noise_size = 0;  
for i = 1:size
    str = input_txt{i};
    str = strsplit(str,' ');
    raw_model =pcread(['.',str{1}]);
    model1 =pcread(['.',str{2}]);
    model2 =pcread(['.',str{3}]);
    type = str{5};
    if strcmp(type, 'com&down')
        com_dow_size =com_dow_size+1;
    else
        noise_size =noise_size + 1;
    end
    raw_model.Normal=pcnormals(raw_model);
    model1.Normal=pcnormals(model1);
    model2.Normal=pcnormals(model2);
    [asBA1, asAB1, asSym1] = point_to_point(raw_model, model1, 'max');
    [asBA2, asAB2, asSym2] = point_to_point(raw_model, model2, 'max');
    if asSym1 < asSym2
        if strcmp(type, 'com&down')
            correct_number_com_dow=correct_number_com_dow+1;
        else
            correct_number_noise=correct_number_noise+1;
        end
    end
end
accuracy1_po2po_Haus_com_dow = correct_number_com_dow/com_dow_size;
accuracy2_po2po_Haus_noise = correct_number_noise/noise_size;
accuracy_po2po_Haus = (correct_number_com_dow+correct_number_noise)/size;


correct_number_com_dow = 0;
correct_number_noise = 0;
com_dow_size = 0;
noise_size = 0;  
for i = 1:size
    str = input_txt{i};
    str = strsplit(str,' ');
    raw_model =pcread(['.',str{1}]);
    model1 =pcread(['.',str{2}]);
    model2 =pcread(['.',str{3}]);
    type = str{5};
    if strcmp(type, 'com&down')
        com_dow_size =com_dow_size+1;
    else
        noise_size =noise_size + 1;
    end
    raw_model.Normal=pcnormals(raw_model);
    model1.Normal=pcnormals(model1);
    model2.Normal=pcnormals(model2);
    [asBA1, asAB1, asSym1] = point_to_point(raw_model, model1, 'PSNR_MSE');
    [asBA2, asAB2, asSym2] = point_to_point(raw_model, model2, 'PSNR_MSE');
    if asSym1 < asSym2
        if strcmp(type, 'com&down')
            correct_number_com_dow=correct_number_com_dow+1;
        else
            correct_number_noise=correct_number_noise+1;
        end
    end
end
accuracy1_po2po_PSNR_MSE_com_dow = correct_number_com_dow/com_dow_size;
accuracy2_po2po_PSNR_MSE_noise = correct_number_noise/noise_size;
accuracy_po2po_PSNR_MSE = (correct_number_com_dow+correct_number_noise)/size;


correct_number_com_dow = 0;
correct_number_noise = 0;
com_dow_size = 0;
noise_size = 0;  
for i = 1:size
    str = input_txt{i};
    str = strsplit(str,' ');
    raw_model =pcread(['.',str{1}]);
    model1 =pcread(['.',str{2}]);
    model2 =pcread(['.',str{3}]);
    type = str{5};
    if strcmp(type, 'com&down')
        com_dow_size =com_dow_size+1;
    else
        noise_size =noise_size + 1;
    end
    raw_model.Normal=pcnormals(raw_model);
    model1.Normal=pcnormals(model1);
    model2.Normal=pcnormals(model2);
    [asBA1, asAB1, asSym1] = point_to_plane(raw_model, model1, 'mse');
    [asBA2, asAB2, asSym2] = point_to_plane(raw_model, model2, 'mse');
    if asSym1 < asSym2
        if strcmp(type, 'com&down')
            correct_number_com_dow=correct_number_com_dow+1;
        else
            correct_number_noise=correct_number_noise+1;
        end
    end
end
accuracy1_po2pl_mse_com_dow = correct_number_com_dow/com_dow_size;
accuracy2_po2pl_mse_noise = correct_number_noise/noise_size;
accuracy_po2pl_mse = (correct_number_com_dow+correct_number_noise)/size;

correct_number_com_dow = 0;
correct_number_noise = 0;
com_dow_size = 0;
noise_size = 0;  
for i = 1:size
    str = input_txt{i};
    str = strsplit(str,' ');
    raw_model =pcread(['.',str{1}]);
    model1 =pcread(['.',str{2}]);
    model2 =pcread(['.',str{3}]);
    type = str{5};
    if strcmp(type, 'com&down')
        com_dow_size =com_dow_size+1;
    else
        noise_size =noise_size + 1;
    end
    raw_model.Normal=pcnormals(raw_model);
    model1.Normal=pcnormals(model1);
    model2.Normal=pcnormals(model2);
    [asBA1, asAB1, asSym1] = point_to_plane(raw_model, model1, 'max');
    [asBA2, asAB2, asSym2] = point_to_plane(raw_model, model2, 'max');
    if asSym1 < asSym2
        if strcmp(type, 'com&down')
            correct_number_com_dow=correct_number_com_dow+1;
        else
            correct_number_noise=correct_number_noise+1;
        end
    end
end
accuracy1_po2pl_Haus_com_dow = correct_number_com_dow/com_dow_size;
accuracy2_po2pl_Haus_noise = correct_number_noise/noise_size;
accuracy_po2pl_Haus = (correct_number_com_dow+correct_number_noise)/size;


correct_number_com_dow = 0;
correct_number_noise = 0;
com_dow_size = 0;
noise_size = 0;  
for i = 1:size
    str = input_txt{i};
    str = strsplit(str,' ');
    raw_model =pcread(['.',str{1}]);
    model1 =pcread(['.',str{2}]);
    model2 =pcread(['.',str{3}]);
    type = str{5};
    if strcmp(type, 'com&down')
        com_dow_size =com_dow_size+1;
    else
        noise_size =noise_size + 1;
    end
    raw_model.Normal=pcnormals(raw_model);
    model1.Normal=pcnormals(model1);
    model2.Normal=pcnormals(model2);
    [asBA1, asAB1, asSym1] = point_to_plane(raw_model, model1, 'PSNR_MSE');
    [asBA2, asAB2, asSym2] = point_to_plane(raw_model, model2, 'PSNR_MSE');
    if asSym1 < asSym2
        if strcmp(type, 'com&down')
            correct_number_com_dow=correct_number_com_dow+1;
        else
            correct_number_noise=correct_number_noise+1;
        end
    end
end
accuracy1_po2pl_PSNR_MSE_com_dow = correct_number_com_dow/com_dow_size;
accuracy2_po2pl_PSNR_MSE_noise = correct_number_noise/noise_size;
accuracy_po2pl_PSNR_MSE = (correct_number_com_dow+correct_number_noise)/size;


correct_number_com_dow = 0;
correct_number_noise = 0;
com_dow_size = 0;
noise_size = 0;  
for i = 1:size
    str = input_txt{i};
    str = strsplit(str,' ');
    raw_model =pcread(['.',str{1}]);
    model1 =pcread(['.',str{2}]);
    model2 =pcread(['.',str{3}]);
    type = str{5};
    if strcmp(type, 'com&down')
        com_dow_size =com_dow_size+1;
    else
        noise_size =noise_size + 1;
    end
    raw_model.Normal=pcnormals(raw_model);
    model1.Normal=pcnormals(model1);
    model2.Normal=pcnormals(model2);
    [asBA1, asAB1, asSym1] = angularSimilarity(raw_model, model1, 'mse');
    [asBA2, asAB2, asSym2] = angularSimilarity(raw_model, model2, 'mse');
    if asSym1 < asSym2
        if strcmp(type, 'com&down')
            correct_number_com_dow=correct_number_com_dow+1;
        else
            correct_number_noise=correct_number_noise+1;
        end
    end
end
accuracy1_pl2pl_mse_com_dow = correct_number_com_dow/com_dow_size;
accuracy2_pl2pl_mse_noise = correct_number_noise/noise_size;
accuracy_pl2pl_mse = (correct_number_com_dow+correct_number_noise)/size;

correct_number_com_dow = 0;
correct_number_noise = 0;
com_dow_size = 0;
noise_size = 0;  
for i = 1:size
    str = input_txt{i};
    str = strsplit(str,' ');
    raw_model =pcread(['.',str{1}]);
    model1 =pcread(['.',str{2}]);
    model2 =pcread(['.',str{3}]);
    type = str{5};
    if strcmp(type, 'com&down')
        com_dow_size =com_dow_size+1;
    else
        noise_size =noise_size + 1;
    end
    raw_model.Normal=pcnormals(raw_model);
    model1.Normal=pcnormals(model1);
    model2.Normal=pcnormals(model2);
    [asBA1, asAB1, asSym1] =angularSimilarity(raw_model, model1, 'max');
    [asBA2, asAB2, asSym2] =angularSimilarity(raw_model, model2, 'max');
    if asSym1 < asSym2
        if strcmp(type, 'com&down')
            correct_number_com_dow=correct_number_com_dow+1;
        else
            correct_number_noise=correct_number_noise+1;
        end
    end
end
accuracy1_pl2pl_Haus_com_dow = correct_number_com_dow/com_dow_size;
accuracy2_pl2pl_Haus_noise = correct_number_noise/noise_size;
accuracy_pl2pl_Haus = (correct_number_com_dow+correct_number_noise)/size;


correct_number_com_dow = 0;
correct_number_noise = 0;
com_dow_size = 0;
noise_size = 0;  
for i = 1:size
    str = input_txt{i};
    str = strsplit(str,' ');
    raw_model =pcread(['.',str{1}]);
    model1 =pcread(['.',str{2}]);
    model2 =pcread(['.',str{3}]);
    type = str{5};
    if strcmp(type, 'com&down')
        com_dow_size =com_dow_size+1;
    else
        noise_size =noise_size + 1;
    end
    raw_model.Normal=pcnormals(raw_model);
    model1.Normal=pcnormals(model1);
    model2.Normal=pcnormals(model2);
    [asBA1, asAB1, asSym1] = angularSimilarity(raw_model, model1, 'PSNR_MSE');
    [asBA2, asAB2, asSym2] = angularSimilarity(raw_model, model2, 'PSNR_MSE');
    if asSym1 < asSym2
        if strcmp(type, 'com&down')
            correct_number_com_dow=correct_number_com_dow+1;
        else
            correct_number_noise=correct_number_noise+1;
        end
    end
end
accuracy1_pl2pl_PSNR_MSE_com_dow = correct_number_com_dow/com_dow_size;
accuracy2_pl2pl_PSNR_MSE_noise = correct_number_noise/noise_size;
accuracy_pl2pl_PSNR_MSE = (correct_number_com_dow+correct_number_noise)/size;