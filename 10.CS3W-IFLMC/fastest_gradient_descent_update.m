function [n,list_t,list_x1,list_x2] = fastest_gradient_descent_update(fx,x10,x20,error)
% 加速梯度下降算法求二元函数的的全局极小值点
% t代表步长；x1、x2代表变量；n代表迭代次数
 
syms x1 x2 t
format short


g = [diff(fx,x1),diff(fx,x2)];   % 求梯度
v = [x1,x2];
 
v0 = [x10,x20];
g0 = subs(g,v,v0);          % 求初始点[x0,y0]的梯度值; subs: Symbolic substitution
 
k = 0;
n = 0;                         % 迭代次数
list_t = [];
list_x1 = [];
list_x2 = [];
 
% 将每一次迭代的变量,分别保存到list_x1.list_x2
list_x1(end+1) = v0(1);
list_x2(end+1) = v0(2);
temp = norm(g0);
while not (temp < error)  
    if k == 2
        % 下降方向
        l_1 = length(list_x1);
        l_2 = length(list_x2);
        x1 = [list_x1(l_1 - 2),list_x2(l_2 - 2)];
        x2 = [list_x1(l_1),list_x2(l_2)];
        d = x2 - x1;      
 
        k = 0;
    else
        % 最速下降方向
        d = - g0;      
 
        k = k + 1;
    end
    % 利用精确一维搜索得到最佳步长
    ft = subs(fx,v,v0 + t * d);
    dft = diff(ft,t);
    t_value = double(solve(dft));
    list_t(end+1) = t_value;      % 记录最佳步长
    
    % 更新并进行准备下一次迭代
    v0 = v0 + t_value * d;        % 求下一个迭代点
    g0 = subs(g,v,v0);      % 下一个迭代点的梯度
    temp = norm(g0);        % 向量范数和矩阵范数
        
    % 将每一次迭代的变量,分别保存到list_x1.list_x2
    list_x1(end+1) = v0(1);
    list_x2(end+1) = v0(2);
        
    % 记录迭代次数
    n = n + 1; 
end
end