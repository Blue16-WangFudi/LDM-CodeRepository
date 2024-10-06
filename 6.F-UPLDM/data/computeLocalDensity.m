% 自定义的计算局部密度函数
function densities = computeLocalDensity(data, epsilon)
    [numData, numFeatures] = size(data);
    densities = zeros(numData, 1);

    for i = 1:numData
        % 计算数据点i到所有其他点的距离
        distances = sqrt(sum((data - data(i, :)).^2, 2));
        
        % 计算在半径epsilon范围内的邻域点数
        numNeighbors = sum(distances < epsilon) - 1; % 减去点自身
        
        % 计算局部密度
        densities(i) = numNeighbors / (pi * epsilon^2);
    end
end
