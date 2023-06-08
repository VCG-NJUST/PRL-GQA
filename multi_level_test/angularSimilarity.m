function [asBA, asAB, asSym] = angularSimilarity(A, B, ERRORTYPE)
%
% Author:
%   Evangelos Alexiou (evangelos.alexiou@epfl.ch)
%
% Reference:
%   E. Alexiou and T. Ebrahimi, "Point Cloud Quality Assessment Metric
%   Based on Angular Similarity," 2018 IEEE International Conference on
%   Multimedia and Expo (ICME), San Diego, CA, 2018, pp. 1-6.
%
%
% angularSimilarity reflects the different orientation of tangent planes
%   that correspond to pairs of points, which are associated as nearest
%   neighbors between models A and B.
%
%   [asBA, asAB, asSym] = angularSimilarity(A, B, ERRORTYPE)
%
%   INPUTS
%       A, B: Point clouds in pointCloud format as defined in MATLAB. The
%       coordinates (e.g., A.Location) and the normal vectors (e.g.,
%       A.Normal) of the input point clouds A and B must exist.
%       ERRORTYPE: Defines how the total similarity is computed, based on
%       the functions 'min', 'mean', 'median', 'max', 'rms', or 'mse'.
%
%   OUTPUTS
%       asBA: The total similarity obtained with point cloud A being the
%       reference. The computed value depends on ERRORTYPE selection.
%       asAB: The total similarity obtained with point cloud B being the
%       reference. The computed value depends on ERRORTYPE selection.
%       asSym: The symmetric total similarity. The computed value depends
%       on ERRORTYPE selection.
%
%   Example:
%   [asBA, asAB, asSym] = angularSimilarity(A, B, 'mse')



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

if isempty(A.Location) || isempty(B.Location)
    error('No coordinates found in input point cloud(s).');
end

if isempty(A.Normal) || isempty(B.Normal)
    error('No normal vectors found in input point cloud(s).');
end   %法线和坐标不能为空

Px=(max([A.Location(:,1);B.Location(:,1)])-min([A.Location(:,1);B.Location(:,1)]))*(max([A.Location(:,1);B.Location(:,1)])-min([A.Location(:,1);B.Location(:,1)]));
Py=(max([A.Location(:,2);B.Location(:,2)])-min([A.Location(:,2);B.Location(:,2)]))*(max([A.Location(:,2);B.Location(:,2)])-min([A.Location(:,2);B.Location(:,2)]));
Pz=(max([A.Location(:,3);B.Location(:,3)])-min([A.Location(:,3);B.Location(:,3)]))*(max([A.Location(:,3);B.Location(:,3)])-min([A.Location(:,3);B.Location(:,3)]));
P_square=Px+Py+Pz;
% Set A as the reference. Loop over B and find nearest neighbor in A
[n1, ~] = knnsearch(A.Location, B.Location);          %Knnsearch（x,y)返回x中每一个y点的最近邻，返回参数1为X中对应序号，参数2为最近邻距离，K默认为1
as_BA = 1 - 2*acos(abs( sum(A.Normal(n1,:).*B.Normal,2)./(sqrt(sum(A.Normal(n1,:).^2,2)).*sqrt(sum(B.Normal.^2,2))) ))/pi;

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

% Set B as the reference. Loop over A and find nearest neighbor in B
[n2, ~] = knnsearch(B.Location, A.Location);
as_AB = 1 - 2*acos(abs( sum(A.Normal.*B.Normal(n2,:),2)./(sqrt(sum(A.Normal.^2,2)).*sqrt(sum(B.Normal(n2,:).^2,2))) ))/pi;

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

% Symmetric total similarity

if strcmp(ERRORTYPE, 'PSNR_MSE')
    asSym = nanmax(asBA, asAB);
else
    asSym = nanmin(asBA, asAB);
end
