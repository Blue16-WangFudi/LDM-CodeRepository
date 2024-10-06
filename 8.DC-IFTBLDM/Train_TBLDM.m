function Outs_Train = Train_TBLDM(A,y_A ,B, y_B,Samples_Train, lambda1,lambda2, C_s, Kernel, QPPs_Solver)

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
   lambda3 = lambda1;
   lambda4 = lambda2;
   l2 = size(B,1); 
   l1 = size(A,1);
   Q2 = (l2*ones(l2)-y_B*y_B')/l2^2;
   Q1 = (l1*ones(l1)-y_A*y_A')/l1^2;
   e2 = ones(l2,1);
   e1 = ones(l1,1);
   
   C1 = C_s.C1;
   C3 = C_s.C3;
   C2 = C_s.C2;
   C4 = C_s.C4;


   m = size(Samples_Train,1);
   CR = 1e-7;
   K = Function_Kernel(Samples_Train, Samples_Train, Kernel);
   K_A = Function_Kernel(A, Samples_Train, Kernel);
   K_B = Function_Kernel(B, Samples_Train, Kernel);


           
   % Parameters for quadprog  
  switch QPPs_Solver
       case 'QP_Matlab'    
           Options.LargeScale = 'off';
           Options.Display = 'off';
           Options.Algorithm =  'trust-region-reflective'; 
           G1 = C1*K + K_A'*K_A + 2*lambda2*K_B'*Q2*K_B + CR*eye(m);
           G2 = C2*K + K_B'*K_B + 2*lambda4*K_A'*Q1*K_A + CR*eye(m);
           alpha1_0 = zeros(l2, 1);
           alpha2_0 = zeros(l1, 1);

           % solver
           H1 = K_B/G1*K_B';
           H1=(H1+H1')/2;
           H2 = K_A/G2*K_A';
           H2=(H2+H2')/2;


           alpha1 = quadprog(H1, -((lambda1/l2)*H1*y_B + e2), [], [], [], [], zeros(l2, 1), C3*e2,alpha1_0,Options);
           beta1 = G1\((lambda1/l2)*K_B'*y_B-K_B'*alpha1);
           alpha2 = quadprog(H2, (lambda3/l1)*H2*y_A - e1, [], [], [], [], zeros(l1, 1), C4*e1,alpha2_0,Options);
           beta2 = G2\((lambda3/l1)*K_A'*y_A-K_A'*alpha2);
       otherwise
           disp('Wrong QPPs_Solver is provided, and we use ''coordinate descent method'' insdead. ')
  end

  
    Outs_Train.alpha1 = alpha1;
    Outs_Train.beta1 = beta1;
    Outs_Train.alpha1 = alpha2;
    Outs_Train.beta2 = beta2;
    Outs_Train.Kernel = Kernel;
    Outs_Train.K = K;
end

