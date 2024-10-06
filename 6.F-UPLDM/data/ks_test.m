function [h,p] = ks_test(data,labels)


% 分离两个类别的数据
class1Data = data(labels == 1, :); % 假设类别1的标签是1
class2Data = data(labels == -1, :); % 假设类别2的标签是2

% 执行 K-S 检验
[h, p, ks2stat] = kstest2(class1Data(:), class2Data(:));

% 打印 K-S 检验结果
fprintf('K-S Statistic: %f\n', ks2stat);
fprintf('P-Value: %f\n', p);

% 判断统计显著性
alpha = 0.05; % 显著性水平
if p < alpha
    fprintf('差异显著，拒绝原假设\n');
else
    fprintf('差异不显著，无法拒绝原假设\n');
end

end

