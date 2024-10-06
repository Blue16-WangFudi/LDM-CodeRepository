function s = IF_new_new(Data,Label,Kernel,u)
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
   P_Ker_P = Function_Kernel(Data_Pos, Data_Pos, Kernel);
   P_Ker_N = Function_Kernel(Data_Pos, Data_Neg, Kernel);
   N_Ker_N = Function_Kernel(Data_Neg, Data_Neg, Kernel);
   
   
   C_pn = sqrt((e_Pos'*P_Ker_P*e_Pos)/(N_Pos^2) + (e_Neg'*N_Ker_N*e_Neg)/(N_Neg^2) - 2*(e_Pos'*P_Ker_N*e_Neg)/(N_Pos*N_Neg)); % p_i（正）与正中心之间的距离
   P_P = sqrt(diag(P_Ker_P)-2*P_Ker_P*e_Pos/N_Pos+(e_Pos'*P_Ker_P*e_Pos)*e_Pos/(N_Pos^2));   % p_i（正）与正中心之间的距离
   r_Pos = max(P_P);
%    delta_Pos = 0.1*r_s;
   P_N = sqrt(diag(P_Ker_P)-2*P_Ker_N*e_Neg/N_Neg+(e_Neg'*N_Ker_N*e_Neg)*e_Pos/(N_Neg^2));   % p_i（正）与负中心之间的距离
   
   N_N = sqrt(diag(N_Ker_N)-2*N_Ker_N*e_Neg/N_Neg+(e_Neg'*N_Ker_N*e_Neg)*e_Neg/(N_Neg^2));   % n_i（负）与负中心之间的距离
   r_Neg = max(N_N);
%    delta_Neg = 0.1*s_s;
   N_P = sqrt(diag(N_Ker_N)-2*P_Ker_N'*e_Pos/N_Pos+(e_Pos'*P_Ker_P*e_Pos)*e_Neg/(N_Pos^2));  % n_i（负）与正中心之间的距离
   
   % Compute the membership of postive data
%    d_p = P_N - P_P;% 由于三角形任意两边之差大于第三边，C_pn永远是大于等于d_p中的值！！！%补：解决中心连线直线上隶属度一样的问题！！！
%    Nmem1 = (d_p - min(d_p)+10e-7)./((C_pn - min(d_p)));%分子加 10e-7 避免出现 0 
   
     d_p = P_P-P_N;
     Nmem1 = (d_p - min(d_p)+10e-7)./((1 - min(d_p)));
   
   
   % Compute the membership of negative data
%    d_n = N_P - N_N;% 由于三角形任意两边之差大于第三边，C_pn永远是大于等于d_p中的值！！！
%    Nmem2 = (d_n - min(d_n)+10e-7)./((C_pn - min(d_n)));
%    
    d_n = N_N-N_P;
    Nmem2 = (d_n - min(d_n)+10e-7)./((1 - min(d_n)));
   
   
   %%
   % Compute the membership of postive data
   Mem1 = (1-P_P/(r_Pos+1e-7));   
  
 % Compute the membership of negative data
   Mem2 = (1- N_N/(r_Neg+1e-7));
   
   %%
%    % Generate s
%     s1=sqrt((Mem1.^2+Nmem1.^2)./2);
%     s2=sqrt((Mem2.^2+Nmem2.^2)./2);

      s1=sqrt((Mem1.^2+(1-Nmem1).^2)./2);
      s2=sqrt((Mem2.^2+(1-Nmem2).^2)./2);
   
   s(Label==1) = s1; %正样本模糊隶属度
   s(Label==-1) = s2; %负样本模糊隶属度
   
   
%  % Compute the membership of postive data
%    s_Pos = zeros(N_Pos, 1);
%    s_Pos(P_P>=P_N) = u*(1-sqrt(P_P(P_P>=P_N)/(r_Pos+delta_Pos)));   
%    s_Pos(~(P_P>=P_N)) = (1-u)*(1-sqrt(P_P(~(P_P>=P_N))/(r_Pos+delta_Pos)));
%   
%  % Compute the membership of negative data
%    s_Neg = zeros(N_Neg, 1);
%    s_Neg(N_N>=N_P) = u*(1-sqrt(  N_N(N_N>=N_P)  /(r_Neg+delta_Neg)));
%    s_Neg(~(N_N>=N_P)) = (1-u)*(1-sqrt(N_N(~(N_N>=N_P))/(r_Neg+delta_Neg)));
%    
%  % Generate s
%    s(Label==1) = s_Pos; %正样本模糊隶属度
%    s(Label==-1) = s_Neg; %负样本模糊隶属度
   
   
%    Mem1=e_Pos-P_P/(r_s+10e-7);%隶属度
%    Nmem1=P_P./(P_N+P_P);%非隶属度
% 
%     Mem2=e_Neg-N_N/(s_s+10e-7);%隶属度
%     Nmem2=N_N./(N_P+N_N);%非隶属度
%     
%     s1=sqrt((Mem1.^2+(e_Pos-Nmem1).^2)./2);
%     s2=sqrt((Mem2.^2+(e_Neg-Nmem2).^2)./2);
    
% % %     加u，处理边际噪声
%     s1(P_P>=P_N) = u*s1(P_P>=P_N);
%     s1(P_P<P_N) = 1*s1(P_P<P_N);
% 
%     s2(N_N>=N_P) = u*s2(N_N>=N_P);
%     s2(N_N<N_P) = 1*s2(N_N<N_P);

%    s.s1 = s1; %正样本模糊隶属度
%    s.s2 = s2; %负样本模糊隶属度
end

