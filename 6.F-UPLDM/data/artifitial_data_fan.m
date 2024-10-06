% 读取Excel文件
file = 'Data/fan_data.xlsx';
data = xlsread(file);

% 提取特征和标签
features = data(:, 1:2);
labels = data(:, 3);

% 可视化数据
scatter(features(labels == 1, 1), features(labels == 1, 2), 'r', 'filled');
hold on;
scatter(features(labels == -1, 1), features(labels == -1, 2), 'b', 'filled');
xlabel('特征 1');
ylabel('特征 2');
legend('+1 类', '-1 类');
title('数据可视化');

% 保存数据为MAT文件
save('Output/fan_data.mat', 'data');
