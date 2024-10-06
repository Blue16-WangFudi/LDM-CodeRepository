function Outs_Train = Train_LDM(Samples_Train, Labels_Train, lambda1, lambda2, C, Kernel, QPPs_Solver)

% Function:  Training of the LDM
%------------------- Input -------------------%
% Samples_Train：  the samples  样本
%  Labels_Train:   the cooresponding labels of Samples  样本的对应标签
%       lambda1:   the parameter for margin std  方差的参数
%       lambda2:   the parameter for margin mean  距离平均值的参数
%             C:   parameter for slack variables   松弛变量的参数
%        Kernel:   kernel type: 'Linear', 'RBF',....  核函数类型
%   QPPs_Solver:    original and dual problem solvers  对偶问题解决器

%------------------- Output -------------------%
% Outs_Train includes:
%    1.Outs_Train.beta      the solution to dual problem  解决对偶问题的方法
%   2.Outs_Train.alpha      the solution to primal problem 解决原始问题的方法
% 3.Outs_Train.Samples      the samples  采样样本
%  4.Outs_Train.Labels      the cooresponding labels of Samples  标记样本的对应标签
%       5.Outs_Train.C      the parameter for slack variables  松弛变量的参数
%     6.Outs_Train.Ker      kernel type  核函数类型

% Author: Wendong Wang
%  Email: d.sylan@foxmail.com
%   data: 2015,9,8



   rand('state', 2015)
   randn('state', 2015)
   

%% Main 
   m = size(Samples_Train, 1);    % The number of samples
   e = ones(m, 1);
   G = Function_Kernel(Samples_Train, Samples_Train, Kernel); %内积
   GY = G*diag(Labels_Train);
   CR = 1e-7;
   Q = 4*lambda1*(m*G'*G-(GY*e)*(GY*e)')/(m^2) + G + CR*eye(m);
   C = C*length(Labels_Train);
   switch QPPs_Solver
       case 'QP_Matlab' 
%            H = GY'*(Q\GY);
           
           H = GY'*inv(Q)*GY;
           H = (H+H')/2;
           z = lambda2*H*e/m-e;
         % Parameters for quadprog         
           beta0 = zeros(m, 1);
           Options.LargeScale = 'off';
           Options.Display = 'off';
           Options.Algorithm = 'interior-point-convex'; 
         % solver
           beta = quadprog(H, z, Labels_Train', -lambda2*e'*Labels_Train/m, [], [], zeros(m, 1), C, beta0, Options);
%            alpha = Q\(GY*(lambda2*e/m+beta));
           alpha = Q\GY*(lambda2*e/m+beta);
       case 'CD_LDM'
           alpha = CD_LDM(Q, GY, lambda2, C);
       otherwise
           disp('Wrong QPPs_Solver is provided, and we use ''coordinate descent method'' insdead. ')
           
           alpha = CD_LDM(Q, GY, lambda2, C);
   end
   Outs_Train.alpha = alpha;
   Outs_Train.Samples = Samples_Train;
   Outs_Train.G = G;
   Outs_Train.Labels = Labels_Train;
   Outs_Train.Ker = Kernel;
   
end

