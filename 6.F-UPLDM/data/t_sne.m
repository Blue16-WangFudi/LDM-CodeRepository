function [reduced_data] = t_sne(Data_feature,Data_label,index)
    % t-SNE降维
    rng('default'); % 设置随机种子以确保结果的可重复性
    options = statset('Display','iter');
    reduced_data = tsne(Data_feature, 'Algorithm', 'barneshut', 'Options', options);
    

    % 可视化t-SNE结果
    gscatter(reduced_data(:, 1), reduced_data(:, 2), Data_label);
    
    % 为每个数据集创建一个唯一的文件名
    filename = sprintf('Data_%d_tSNE.pdf', index);

    % 指定保存路径，将文件保存为PDF
    saveas(gcf, fullfile('C:\Users\63208\Desktop\董师兄第三次返修\数据分析', filename), 'pdf');
    
    % 关闭当前图形以便下一轮迭代
    close(gcf);
end

