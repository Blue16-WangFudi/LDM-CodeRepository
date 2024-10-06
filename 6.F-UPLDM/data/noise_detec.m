function [classLabel,numOutliers] = noise_detec(Data_feature,Data_label,index)
    %%异常值检测
    % 导入包含标签的数据集
    data = [Data_feature Data_label]; % 替换为你的数据集文件名和路径
    labels = data(:, end); % 提取最后一列作为标签
    data = data(:, 1:end-1); % 去除最后一列，保留特征

    % 分别检测每个类别中的异常值
    uniqueLabels = unique(labels); % 获取唯一的标签值

    for classLabel = uniqueLabels'
        % 提取属于当前类别的数据
        classData = data(labels == classLabel, :);

        % 绘制箱线图
        subplot(1, numel(uniqueLabels), find(uniqueLabels == classLabel));
        boxplot(classData, 'whisker', 1.5);

        % 添加标签和标题
        xlabel('Feature');
        ylabel('Data Values');
        title(['Boxplot for Class ', num2str(classLabel)]);

        % 检测异常值
        outliers = isoutlier(classData, 'quartiles'); % 使用四分位距法检测异常值
        numOutliers = sum(outliers(:));
        fprintf('类别 %d 中的异常值数量：%d\n', classLabel, numOutliers);
    end
    
    % 为每个数据集创建一个唯一的文件名
    filename = sprintf('Data_%d_tSNE.pdf', index);

    % 指定保存路径，将文件保存为PDF
    saveas(gcf, fullfile('C:\Users\63208\Desktop\董师兄第三次返修\异常检测', filename), 'pdf');
    
    % 关闭当前图形以便下一轮迭代
    close(gcf);
end

