function Outputs = Train_SVM(Samples_Train, Labels_Train, C, Kernel, QPPs_Solver)

% Function:  Training of the SVM
%------------------- Input -------------------%
% Samples_Train：  the samples  训练样本
%  Labels_Train:   the cooresponding labels of the samples  训练样本的标签
%             C:   parameter for slack variables  松弛变量的参数
%        Kernel:   kernel type: 'Linear', 'RBF',....  核函数
%   QPPs_Solver:   original and dual problem solvers  原始问题和对偶问题

%------------------- Output -------------------%
% Outs_Train includes:
%    1.Outs_Train.beta      the solution to dual problem  对偶问题的解：β
%       2.Outs_Train.Q      Q=Y*X'*X*Y
% 3.Outs_Train.Samples      the samples  样本
%  4.Outs_Train.Labels      the cooresponding labels of Samples  标签
%       5.Outs_Train.C      the parameter for slack variables  松弛变量的参数
%     6.Outs_Train.Ker      kernel type  核函数类型

% Author: Wendong Wang
%  Email: d.sylan@foxmail.com
%   data: 2015,9,8 

%    rng('default') 
   
%%%********************************************************************************************************************？？？怎么训练？
%% Main 
   m = size(Samples_Train, 1);    % The number of samples样本个数
   e = ones(m, 1);
   if strcmp(Kernel.Type, 'Linear')
       Q = (Labels_Train*Labels_Train').*(Samples_Train*Samples_Train');  %核函数类型为Linear核函数,对x(1)和x(2)使用核函数，即Samples_Train(:,1)和Samples_Train(:,2)，得到高维样本点
   else
       Q = (Labels_Train*Labels_Train').*Function_Kernel(Samples_Train, Samples_Train, Kernel);%否则使用其他核函数,对x(1)和x(2)使用核函数，即Samples_Train(:,1)和Samples_Train(:,2)，得到高维样本点
   end
   Q = (Q + Q')/2;%%********************************************************************************************************************？？？？？？
   CR = 1e-7;
   Q = Q + CR*eye(m);%%********************************************************************************************************************？？？CR是什么??
   switch QPPs_Solver
       
       case 'qp'  
           if strcmp(Kernel.Type, 'Linear')
               neqcstr = 1;   % For Linear 使用Linear核函数
           else
               neqcstr = 0;   % For RBF  使用径向基核函数
           end
          beta0 = zeros(m, 1);
          beta = qp(Q, -e, Labels_Train', 0, zeros(m, 1), C, beta0, neqcstr);%%**************************************************************************************？？？qp文件?
           
       case 'QP_Matlab'
           beta0 = zeros(m, 1);
           Options.LargeScale = 'off';
           Options.Display = 'off';
           Options.Algorithm = 'interior-point-convex';
           beta = quadprog(Q, -e, [], [], Labels_Train', 0, zeros(m, 1), C, beta0, Options);
           
       case 'CD_SVM'
           beta = CD_SVM(Q, Labels_Train, CR, C);
           
       otherwise
           disp('Wrong QPPs_Solver is provided, and insdead we use ''qp''. ')
           if strcmp(Kernel.Type, 'Linear')
               neqcstr = 1;   % For Linear
           else
               neqcstr = 0;   % For RBF
           end
           beta0 = zeros(m, 1);
           beta = qp(Q, -e, Labels_Train', 0, zeros(m, 1), C, beta0, neqcstr);
   end
   Outputs.beta = beta;
   Outputs.Q = Q;
   Outputs.Samples = Samples_Train;
   Outputs.Labels = Labels_Train;
   Outputs.C = C;
   Outputs.Kernel = Kernel;
   
end

