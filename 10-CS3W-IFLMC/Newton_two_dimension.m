function [n,list_x1,list_x2] = Newton_two_dimension(fx,x10,x20,eps)
% 最小梯度下降算法求二元函数的的全局极小值点
% 固定步长t=1；x1、x2代表变量；n代表迭代次数
 
syms x1 x2 d 
format short
 
g = [diff(fx,x1),diff(fx,x2)];   % 求梯度
Hessian = hessian(fx,[x1,x2]);   % 求hessian矩阵
 
v = [x1,x2];
v0 = [x10,x20];
g0 = subs(g,v,v0);          % 求初始点[x0,y0]的梯度值; subs: Symbolic substitution
 
n = 0;                      % 迭代次数
list_x1 = [];
list_x2 = [];
t = 1;
 
% 将每一次迭代的变量,分别保存到list_x1.list_x2
list_x1(end+1) = v0(1);
list_x2(end+1) = v0(2);
temp = norm(g0);
while not (temp < eps)
    eigValue = eig(subs(Hessian,v,v0));
    if all(eigValue) > 0        
        % 下降方向
        d = - inv(subs(Hessian,v,v0)) * subs(g,v,v0)';
    
        % 更新
        v0 = v0 + t * d'; 
        % 将每一次迭代的变量,分别保存到list_x1.list_x2
        list_x1(end+1) = v0(1);
        list_x2(end+1) = v0(2);    
    
        % 准备下一次迭代
        g0 = subs(g,v,v0);
        temp = norm(g0);
 
        n = n + 1;
    else
        disp("Hessian矩阵不是正定矩阵");
    end
end
end