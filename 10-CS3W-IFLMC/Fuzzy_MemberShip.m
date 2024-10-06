function s = Fuzzy_MemberShip(Data, Label, Kernel, u)

% The code only fits for the binary classification problem

% Data: The classification data whose samples lie in row

% Label: The label of data

% Some options for the membership degree function


%% Main  
   N_Samples = length(Label);
   s = zeros(N_Samples, 1);
 % Abstract the positive and negative data
   Data_Pos = Data(Label==1, :);
   N_Pos = sum(Label==1);
   e_Pos = ones(N_Pos, 1);
   
   Data_Neg = Data(Label==-1, :); 
   N_Neg = sum(Label==-1);
   e_Neg = ones(N_Neg, 1);
 % Processing
   P_Ker_P = Function_Kernel(Data_Pos, Data_Pos, Kernel);
   P_Ker_N = Function_Kernel(Data_Pos, Data_Neg, Kernel);
   N_Ker_N = Function_Kernel(Data_Neg, Data_Neg, Kernel);
   
   P_P = diag(P_Ker_P)-2*P_Ker_P*e_Pos/N_Pos+(e_Pos'*P_Ker_P*e_Pos)*e_Pos/(N_Pos^2);   % The distance between p_i(positive) and the positive center
   r_Pos = max(P_P);
   delta_Pos = 0.1*r_Pos;
   P_N = diag(P_Ker_P)-2*P_Ker_N*e_Neg/N_Neg+(e_Neg'*N_Ker_N*e_Neg)*e_Pos/(N_Neg^2);   % The distance between p_i(positive) and the positive center
   
   N_N = diag(N_Ker_N)-2*N_Ker_N*e_Neg/N_Neg+(e_Neg'*N_Ker_N*e_Neg)*e_Neg/(N_Neg^2);   % The distance between n_i(negative) and the negative cente
   r_Neg = max(N_N);
   delta_Neg = 0.1*r_Neg;
   N_P = diag(N_Ker_N)-2*P_Ker_N'*e_Pos/N_Pos+(e_Pos'*P_Ker_P*e_Pos)*e_Neg/(N_Pos^2);  % The distance between n_i(negative) and the positive center
   
 % Compute the membership of postive data
   s_Pos = zeros(N_Pos, 1);
   s_Pos(P_P>=P_N) = u*(1-sqrt(P_P(P_P>=P_N)/(r_Pos+delta_Pos)));   
   s_Pos(~(P_P>=P_N)) = (1-u)*(1-sqrt(P_P(~(P_P>=P_N))/(r_Pos+delta_Pos)));
  
 % Compute the membership of negative data
   s_Neg = zeros(N_Neg, 1);
   s_Neg(N_N>=N_P) = u*(1-sqrt(  N_N(N_N>=N_P)  /(r_Neg+delta_Neg)));
   s_Neg(~(N_N>=N_P)) = (1-u)*(1-sqrt(N_N(~(N_N>=N_P))/(r_Neg+delta_Neg)));
   
 % Generate s
   s(Label==1) = s_Pos;
   s(Label==-1) = s_Neg;
   
   
   
   
   
   



end

