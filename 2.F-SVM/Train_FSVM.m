function Outputs = Train_FSVM(Samples_Train, Labels_Train, C, Kernel, QPPs_Solver)

% Function:  Training of the SVM
%------------------- Input -------------------%
% Samples_Train£º  the samples
%  Labels_Train:   the cooresponding labels of the samples
%             C:   parameter for slack variables 
%        Kernel:   kernel type: 'Linear', 'RBF',....
%   QPPs_Solver:    original and dual problem solvers

%------------------- Output -------------------%
% Outs_Train includes:
%    1.Outs_Train.beta      the solution to dual problem
%       2.Outs_Train.Q      Q=Y*X'*X*Y
% 3.Outs_Train.Samples      the samples
%  4.Outs_Train.Labels      the cooresponding labels of Samples
%       5.Outs_Train.C      the parameter for slack variables
%     6.Outs_Train.Ker      kernel type

% Author: Wendong Wang
%  Email: d.sylan@foxmail.com
%   data: 2015,9,8 

   rng('default') 
   

%% Main 
   m = size(Samples_Train, 1);    % The number of samples
   e = ones(m, 1);
   if strcmp(Kernel.Type, 'Linear')
       Q = (Labels_Train*Labels_Train').*(Samples_Train*Samples_Train');
   else
       Q = (Labels_Train*Labels_Train').*Function_Kernel(Samples_Train, Samples_Train, Kernel);
   end
   Q = (Q + Q')/2;
   CR = 1e-7;
   Q = Q + CR*eye(m);
   switch QPPs_Solver
       
       case 'qp'
           if strcmp(Kernel.Type, 'Linear')
               neqcstr = 1;   % For Linear
           else
               neqcstr = 0;   % For RBF
           end
          beta0 = zeros(m, 1);
          beta = qp(Q, -e, Labels_Train', 0, zeros(m, 1), C, beta0, neqcstr);
           
       case 'QP_Matlab'
           beta0 = zeros(m, 1);
           Options.LargeScale = 'off';
           Options.Display = 'off';
           Options.Algorithm = 'active-set';
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

