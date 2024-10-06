function Outputs = Train_SVM(Samples_Train, Labels_Train, C, Kernel, QPPs_Solver)

% Function:  Training of the SVM
%------------------- Input -------------------%
% Samples_Train��  the samples  ѵ������
%  Labels_Train:   the cooresponding labels of the samples  ѵ�������ı�ǩ
%             C:   parameter for slack variables  �ɳڱ����Ĳ���
%        Kernel:   kernel type: 'Linear', 'RBF',....  �˺���
%   QPPs_Solver:   original and dual problem solvers  ԭʼ����Ͷ�ż����

%------------------- Output -------------------%
% Outs_Train includes:
%    1.Outs_Train.beta      the solution to dual problem  ��ż����Ľ⣺��
%       2.Outs_Train.Q      Q=Y*X'*X*Y
% 3.Outs_Train.Samples      the samples  ����
%  4.Outs_Train.Labels      the cooresponding labels of Samples  ��ǩ
%       5.Outs_Train.C      the parameter for slack variables  �ɳڱ����Ĳ���
%     6.Outs_Train.Ker      kernel type  �˺�������

% Author: Wendong Wang
%  Email: d.sylan@foxmail.com
%   data: 2015,9,8 

%    rng('default') 
   
%%%********************************************************************************************************************��������ôѵ����
%% Main 
   m = size(Samples_Train, 1);    % The number of samples��������
   e = ones(m, 1);
   if strcmp(Kernel.Type, 'Linear')
       Q = (Labels_Train*Labels_Train').*(Samples_Train*Samples_Train');  %�˺�������ΪLinear�˺���,��x(1)��x(2)ʹ�ú˺�������Samples_Train(:,1)��Samples_Train(:,2)���õ���ά������
   else
       Q = (Labels_Train*Labels_Train').*Function_Kernel(Samples_Train, Samples_Train, Kernel);%����ʹ�������˺���,��x(1)��x(2)ʹ�ú˺�������Samples_Train(:,1)��Samples_Train(:,2)���õ���ά������
   end
   Q = (Q + Q')/2;%%********************************************************************************************************************������������
   CR = 1e-7;
   Q = Q + CR*eye(m);%%********************************************************************************************************************������CR��ʲô??
   switch QPPs_Solver
       
       case 'qp'  
           if strcmp(Kernel.Type, 'Linear')
               neqcstr = 1;   % For Linear ʹ��Linear�˺���
           else
               neqcstr = 0;   % For RBF  ʹ�þ�����˺���
           end
          beta0 = zeros(m, 1);
          beta = qp(Q, -e, Labels_Train', 0, zeros(m, 1), C, beta0, neqcstr);%%**************************************************************************************������qp�ļ�?
           
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

