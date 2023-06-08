

function [srocc,krocc,plcc,rmse,or] = verify_performance(mos,predict_mos)

predict_mos = predict_mos(:);
mos = mos(:);

%initialize the parameters used by the nonlinear fitting function
beta(1) = 10;
beta(2) = 0;
beta(3) = mean(predict_mos);
beta(4) = 0.1;
beta(5) = 0.1;

%fitting a curve using the data
[bayta,ehat,J] = nlinfit(predict_mos,mos,@logistic,beta);%多元非线性回归，1为自变量，2为因变量，3为函数模型，4为要求系数的初值
%given a ssim value, predict the correspoing mos (ypre) using the fitted curve
[ypre,junk] = nlpredci(@logistic,predict_mos,bayta,ehat,J);%非线性模型置信区间预测
% ypre = predict(logistic,fsimValues,bayta,ehat,J);

N_false=0;
theta=std(mos)
for i=1:length(mos)
    if ypre(i)>(mos(i)+2*theta) || ypre(i)<(mos(i)-2*theta) 
        N_false=N_false+1;
    end
end
N_false


rmse = sqrt(sum((ypre - mos).^2) / length(mos));%root meas squared error
or=N_false/length(mos);
plcc = corr(mos, ypre, 'type','Pearson'); %pearson linear coefficient
srocc = corr(mos, predict_mos, 'type','spearman');
krocc = corr(mos, predict_mos, 'type','Kendall');
end
function yhat = logistic(bayta,X)

bayta1 = bayta(1); 
bayta2 = bayta(2); 
bayta3 = bayta(3); 
bayta4 = bayta(4);
bayta5 = bayta(5);

logisticPart = 0.5 - 1./(1 + exp(bayta2 * (X - bayta3)));

yhat = bayta1 * logisticPart + bayta4*X + bayta5;

end
