function [asBA, asAB, asSym] = point_to_point(A, B, ERRORTYPE)
if nargin < 2         %nargin探测函数参数个数
    error('Too few input arguments.');
elseif nargin==2
    ERRORTYPE = 'mean';
elseif nargin>2
    switch ERRORTYPE
        case {'min', 'mean', 'median', 'max', 'rms', 'mse','PSNR_MSE','PSNR_Hausdorff','PSNR_RMS'}
        otherwise
            error('ERRORTYPE is not supported.');
    end
end

Px=(max([A.Location(:,1);B.Location(:,1)])-min([A.Location(:,1);B.Location(:,1)]))*(max([A.Location(:,1);B.Location(:,1)])-min([A.Location(:,1);B.Location(:,1)]));
Py=(max([A.Location(:,2);B.Location(:,2)])-min([A.Location(:,2);B.Location(:,2)]))*(max([A.Location(:,2);B.Location(:,2)])-min([A.Location(:,2);B.Location(:,2)]));
Pz=(max([A.Location(:,3);B.Location(:,3)])-min([A.Location(:,3);B.Location(:,3)]))*(max([A.Location(:,3);B.Location(:,3)])-min([A.Location(:,3);B.Location(:,3)]));
P_square=Px+Py+Pz;
[n1, ~] = knnsearch(A.Location, B.Location);  
as_BA=sum((A.Location(n1,:)-B.Location).^2,2);
if strcmp(ERRORTYPE, 'mean')
    asBA = nanmean(real(as_BA));  %real 复数实值部分
elseif strcmp(ERRORTYPE, 'min')
    asBA = nanmin(real(as_BA));
elseif strcmp(ERRORTYPE, 'max')
    asBA = nanmax(real(as_BA));
elseif strcmp(ERRORTYPE, 'median')
    asBA = nanmedian(real(as_BA));
elseif strcmp(ERRORTYPE, 'rms')
    asBA = sqrt(nanmean(real(as_BA).^2));
elseif strcmp(ERRORTYPE, 'mse')
    asBA = nanmean(real(as_BA).^2);
elseif strcmp(ERRORTYPE, 'PSNR_MSE')
    asBA =10*log10(P_square/nanmean(real(as_BA).^2));
elseif strcmp(ERRORTYPE, 'PSNR_Hausdorff')
    asBA =10*log10(P_square/nanmax(real(as_BA))); 
elseif strcmp(ERRORTYPE, 'PSNR_RMS')
    asBA =10*log10(P_square/sqrt(nanmean(real(as_BA).^2))); 
else 
    error('WrongInput');
end
[n2, ~] = knnsearch(B.Location, A.Location);
as_AB=sum((B.Location(n2,:)-A.Location).^2,2);
if strcmp(ERRORTYPE, 'mean')
    asAB = nanmean(real(as_AB));
elseif strcmp(ERRORTYPE, 'min')
    asAB = nanmin(real(as_AB));
elseif strcmp(ERRORTYPE, 'max')
    asAB = nanmax(real(as_AB));
elseif strcmp(ERRORTYPE, 'median')
    asAB = nanmedian(real(as_AB));
elseif strcmp(ERRORTYPE, 'rms')
    asAB = sqrt(nanmean(real(as_AB).^2));
elseif strcmp(ERRORTYPE, 'mse')
    asAB = nanmean(real(as_AB).^2);
elseif strcmp(ERRORTYPE, 'PSNR_MSE')
    asAB =10*log10(P_square/nanmean(real(as_AB).^2));
elseif strcmp(ERRORTYPE, 'PSNR_Hausdorff')
    asAB =10*log10(P_square/nanmax(real(as_AB))); 
elseif strcmp(ERRORTYPE, 'PSNR_RMS')
    asAB =10*log10(P_square/sqrt(nanmean(real(as_AB).^2))); 
else
    error(message('WrongInput'));
end

asSym = nanmin(asBA, asAB);