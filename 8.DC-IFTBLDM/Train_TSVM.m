function Outs_Train = Train_TSVM(A,y_A ,B, y_B,Samples_Train, C_s, Kernel, QPPs_Solver)

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
%     6.Outs_Train.Ker      kernel type0
%  7.     Outs_Train.K      K used to compute the margin mean and margin variance
%

% Author: Wendong Wang
%  Email: d.sylan@foxmail.com
%   data: 2015,9,8
% updata: 2015,10,7


%  rand('state', 2015)
%  randn('state', 2015)


%% Main 
   l2 = size(B,1); 
   l1 = size(A,1);
   e2 = ones(l2,1);
   e1 = ones(l1,1);
   
   C1 = C_s.C1;
   C2 = C_s.C2;


   m = size(Samples_Train,1);
   CR = 1e-7;
   K = Function_Kernel(Samples_Train, Samples_Train, Kernel);
   K_A = Function_Kernel(A, Samples_Train, Kernel);
   K_B = Function_Kernel(B, Samples_Train, Kernel);
   P = [K_A e1];
   Q = [K_B e2];


           
   % Parameters for quadprog  
  switch QPPs_Solver
       case 'QP_Matlab'    
           Options.LargeScale = 'off';
           Options.Display = 'off';
           Options.Algorithm =  'trust-region-reflective'; 
           alpha1_0 = zeros(l2, 1);
           alpha2_0 = zeros(l1, 1);

           % solver
           H1 = Q/(P'*P + CR*eye(m+1))*Q';
           H1=(H1+H1')/2;
           H2 = P/(Q'*Q + CR*eye(m+1))*P';
           H2=(H2+H2')/2;


           alpha1 = quadprog(H1, -e2, [], [], [], [], zeros(l2, 1), C1*e2,alpha1_0,Options);
           alpha2 = quadprog(H2,-e1, [], [], [], [], zeros(l1, 1), C2*e1,alpha2_0,Options);
       otherwise
           disp('Wrong QPPs_Solver is provided, and we use ''coordinate descent method'' insdead. ')
  end

  
    Outs_Train.alpha1 = alpha1;
    Outs_Train.alpha2 = alpha2;
    Outs_Train.P = P;
    Outs_Train.Q = Q;
    Outs_Train.m = m;
    Outs_Train.Kernel = Kernel;
    Outs_Train.K = K;
end

