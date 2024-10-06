% 自定义的计算局部异常度函数
function outlierScores = computeOutlierScore(data, densities, minPts,epsilon)
    [numData, numFeatures] = size(data);
    outlierScores = zeros(numData, 1);

    for i = 1:numData
        % 计算数据点i到所有其他点的距离
        distances = sqrt(sum((data - data(i, :)).^2, 2));
        
        % 获取在半径范围内的数据点索引
        neighborIndices = find(distances < epsilon);
        
        % 必须满足最小邻域点数条件
        if numel(neighborIndices) >= minPts
            % 计算邻域内点的局部密度之和
            neighborDensities = densities(neighborIndices);
            neighborDensitySum = sum(neighborDensities);
            
            % 计算局部异常度
            outlierScores(i) = densities(i) / (neighborDensitySum / minPts);
        end
    end
end