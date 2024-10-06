function s = Fuzzy_MemberShip(Data, Label, Kernel, u, p1)

% The code only fits for the binary classification problem  该代码仅适合二分类问题

% Data: The classification data whose samples lie in row  样本位于行中的分类数据

% Label: The label of data

% Some options for the membership degree function  隶属度函数的一些选项


%% Main  
   N_Samples = length(Label);
   s = zeros(N_Samples, 1);
 % Abstract the positive and negative data  提取正负数据
   Data_Pos = Data(Label==1, :);
   N_Pos = sum(Label==1);
   e_Pos = ones(N_Pos, 1);
   
   Data_Neg = Data(Label==-1, :); 
   N_Neg = sum(Label==-1);
   e_Neg = ones(N_Neg, 1);
 % Processing
   P_Ker_P = svkernel( Kernel,Data_Pos, Data_Pos,p1);
   P_Ker_N = svkernel( Kernel,Data_Pos, Data_Neg,p1);
   N_Ker_N = svkernel( Kernel,Data_Neg, Data_Neg,p1);
   
   P_P = diag(P_Ker_P)-2*P_Ker_P*e_Pos/N_Pos+(e_Pos'*P_Ker_P*e_Pos)*e_Pos/(N_Pos^2);   % p_i（正）与正中心之间的距离
   r_Pos = max(P_P);
   delta_Pos = 0.1*r_Pos;
   P_N = diag(P_Ker_P)-2*P_Ker_N*e_Neg/N_Neg+(e_Neg'*N_Ker_N*e_Neg)*e_Pos/(N_Neg^2);   % p_i（正）与负中心之间的距离
   
   N_N = diag(N_Ker_N)-2*N_Ker_N*e_Neg/N_Neg+(e_Neg'*N_Ker_N*e_Neg)*e_Neg/(N_Neg^2);   % n_i（负）与负中心之间的距离
   r_Neg = max(N_N);
   delta_Neg = 0.1*r_Neg;
   N_P = diag(N_Ker_N)-2*P_Ker_N'*e_Pos/N_Pos+(e_Pos'*P_Ker_P*e_Pos)*e_Neg/(N_Pos^2);  % n_i（负）与正中心之间的距离
   
 % Compute the membership of postive data
   s_Pos = zeros(N_Pos, 1);
   s_Pos(P_P>=P_N) = u*(1-sqrt(P_P(P_P>=P_N)/(r_Pos+delta_Pos)));   
   s_Pos(~(P_P>=P_N)) = (1-u)*(1-sqrt(P_P(~(P_P>=P_N))/(r_Pos+delta_Pos)));
  
 % Compute the membership of negative data
   s_Neg = zeros(N_Neg, 1);
   s_Neg(N_N>=N_P) = u*(1-sqrt(  N_N(N_N>=N_P)  /(r_Neg+delta_Neg)));
   s_Neg(~(N_N>=N_P)) = (1-u)*(1-sqrt(N_N(~(N_N>=N_P))/(r_Neg+delta_Neg)));
   
 % Generate s
   s(Label==1) = s_Pos; %正样本模糊隶属度
   s(Label==-1) = s_Neg; %负样本模糊隶属度
   
   
   
   
   
   



end

