function Outs_Train = Train_LDM(Samples_Train, Labels_Train, lambda1, lambda2, C, Kernel, QPPs_Solver)

% Function:  Training of the LDM
%------------------- Input -------------------%
% Samples_Train：  the samples
%  Labels_Train:   the cooresponding labels of Samples
%       lambda1:   the parameter for margin std
%       lambda2:   the parameter for margin mean
%             C:   parameter for slack variables    %松弛变量
%        Kernel:   kernel type: 'Linear', 'RBF',....
%   QPPs_Solver:    original and dual problem solvers

%------------------- Output -------------------%
% Outs_Train includes:
%    1.Outs_Train.beta      the solution to dual problem
%   2.Outs_Train.alpha      the solution to primal problem
% 3.Outs_Train.Samples      the samples
%  4.Outs_Train.Labels      the cooresponding labels of Samples
%       5.Outs_Train.C      the parameter for slack variables
%     6.Outs_Train.Ker      kernel type

% Author: Wendong Wang
%  Email: d.sylan@foxmail.com
%   data: 2015,9,8


   rand('state', 2015)
   randn('state', 2015)
   

%% Main 
   m = size(Samples_Train, 1);    % The number of samples
   e = ones(m, 1);
   G = Function_Kernel(Samples_Train, Samples_Train, Kernel);
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
         % Parameters for quadprog %四驱
           beta0 = zeros(m, 1);
           Options.LargeScale = 'off';
           Options.Display = 'off';
           Options.Algorithm = 'active-set'; 
         % solver
           beta = quadprog(H, z, Labels_Train', -lambda2*e'*Labels_Train/m, [], [], zeros(m, 1), C, beta0, Options);  %二次规划，内置函数求解。标准形式为：min1/2xTHx+fTx
%            alpha = Q\(GY*(lambda2*e/m+beta));
           alpha = inv(Q)*GY*(lambda2*e/m+beta);
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

