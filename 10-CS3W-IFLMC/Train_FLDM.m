function Outs_Train = Train_FLDM(Samples_Train, Labels_Train, lambda1, lambda2, C_s, FLDM_Type, Kernel, QPPs_Solver)

% Function:  Training of the FLDM
%------------------- Input -------------------%
% Samples_Train£º  the samples
%  Labels_Train:   the cooresponding labels of Samples
%       lambda1:   the parameter for margin std
%       lambda2:   the parameter for margin mean
%         F_LDM:   a structure for F1_LDM 
%        Kernel:   kernel type: 'Linear', 'RBF',....
%   QPPs_Solver:    original and dual problem solvers

%------------------- Output -------------------%
% Outs_Train includes:
%    1.Outs_Train.beta      the solution to dual problem
%       2.Outs_Train.u      the solution to dual problem
%       3.Outs_Train.C      the parameter for slack variables
% 4.Outs_Train.Samples      the samples
%  5.Outs_Train.Labels      the cooresponding labels of Samples
%     6.Outs_Train.Ker      kernel type
%  7.     Outs_Train.K      K used to compute the margin mean and margin variance
%

% Author: Wendong Wang
%  Email: d.sylan@foxmail.com
%   data: 2015,9,8
% updata: 2015,10,7


 rand('state', 2015)
 randn('state', 2015)


%% Main 
   [m, n] = size(Samples_Train);   
   e = ones(m, 1);
   C = C_s.C;
   s = C_s.s;

   if strcmp(Kernel.Type, 'Linear')
       K = [Samples_Train, e]';
       E = eye(n+1);       
   else
       K = [Function_Kernel(Samples_Train, Samples_Train, Kernel), e]';
       E = blkdiag(Function_Kernel(Samples_Train, Samples_Train, Kernel), 1);
   end
   
   CR = 1e-7;
   if strcmp(FLDM_Type, 'F1_LDM')    
       lambda2_eORs = lambda2*e;
       C_eORs = C.*s;
       K_IorS = K; 
   else
       lambda2_eORs = lambda2*s;     
       C_eORs = C;
       K_IorS = K*diag(s);
   end
   Q = E + 4*lambda1*(m*K_IorS*K_IorS'-K_IorS*Labels_Train*Labels_Train'*K_IorS')/(m^2);
   if strcmp(Kernel.Type, 'Linear')
       Q = Q + CR*eye(n+1);    
   else
       Q = Q + CR*eye(m+1);    
   end
   KY = K*diag(Labels_Train);
   
   switch QPPs_Solver
       
       case 'QP_Matlab'
         % Parameters for quadprog         
           beta0 = zeros(m, 1);
           Options.LargeScale = 'off';
           Options.Display = 'off';
           Options.Algorithm = 'active-set'; 
         % solver
           H = KY'*inv(Q)*KY;
           H = (H+H')/2;
           z = H*lambda2_eORs/m-e;
           beta = quadprog(H, z, [], [], [], [], zeros(m, 1), C_eORs, beta0, Options);
           u = Q\KY*(lambda2_eORs/m+beta);   
           
       case 'CD_FLDM'
           [u, beta] = CD_FLDM(Q, KY, lambda2_eORs, C_eORs);
           
       otherwise
           disp('Wrong QPPs_Solver is provided, and we use ''coordinate descent method'' insdead. ')
           [u, beta] = CD_FLDM(Q, KY, lambda2_eORs, C_eORs);
           
   end
   
   Outs_Train.u = u;
   if strcmp(Kernel.Type, 'Linear')
       Outs_Train.beta = beta;
   end
   Outs_Train.Samples = Samples_Train;
   Outs_Train.K = K;
   Outs_Train.Labels = Labels_Train;
   Outs_Train.Ker = Kernel;
end

