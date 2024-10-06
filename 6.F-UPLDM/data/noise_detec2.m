function [numOutliers] = noise_detec2(data,labels)

% 设置DBSCAN参数
epsilon = 0.5; % 半径范围
minPts = 5; % 最小邻域点数

% 初始化两个变量用于存储两个类别的异常值
outliersClass1 = [];
outliersClass2 = [];

% 分别计算两个类别的异常值
uniqueLabels = unique(labels); % 获取唯一的类别标签

for classLabel = uniqueLabels'
    % 提取属于当前类别的数据
    classData = data(labels == classLabel, :);
    
    % 使用DBSCAN进行异常值检测
    [idx, isOutlier] = dbscan(classData, epsilon, minPts);
    
    % 将异常值标记为1，非异常值标记为0
    outliers = isOutlier;
    
    % 打印异常值的数量
    numOutliers = sum(outliers);
%     fprintf('Class %d - Number of outliers: %d\n', classLabel, numOutliers);
    
    % 将异常值保存到相应的变量中
    if classLabel == 1
        outliersClass1 = [outliersClass1; outliers];
    else
        outliersClass2 = [outliersClass2; outliers];
    end
end

% 打印两个类别的异常值数量
numOutliersClass1 = sum(outliersClass1);
numOutliersClass2 = sum(outliersClass2);
fprintf('Number of outliers in Class 1: %d\n', numOutliersClass1);
fprintf('Number of outliers in Class -1: %d\n', numOutliersClass2);

% 在这里你可以进一步处理异常值，根据需要保存或删除它们


% % 为每个数据集创建一个唯一的文件名
% filename = sprintf('Data_%d_tSNE.pdf', index);
% 
% % 指定保存路径，将文件保存为PDF
% saveas(gcf, fullfile('C:\Users\63208\Desktop\董师兄第三次返修\异常检测', filename), 'pdf');
% 
% % 关闭当前图形以便下一轮迭代
% close(gcf);

end

