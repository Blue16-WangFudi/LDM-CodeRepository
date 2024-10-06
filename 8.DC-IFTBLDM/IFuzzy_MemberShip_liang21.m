function s = IFuzzy_MemberShip_liang21(Data, Label, Kernel, u)

% The code only fits for the binary classification problem  �ô�����ʺ϶���������

% Data: The classification data whose samples lie in row  ����λ�����еķ�������

% Label: The label of data

% Some options for the membership degree function  �����Ⱥ�����һЩѡ��


%% Main  
   N_Samples = length(Label);
%    s = zeros(N_Samples, 1);
 % Abstract the positive and negative data  ��ȡ��������
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
   
   P_P = sqrt(diag(P_Ker_P)-2*P_Ker_P*e_Pos/N_Pos+(e_Pos'*P_Ker_P*e_Pos)*e_Pos/(N_Pos^2));   % p_i��������������֮��ľ���
   f_s = max(P_P);
   delta_Pos = 0.1*f_s;
   P_N = sqrt(diag(P_Ker_P)-2*P_Ker_N*e_Neg/N_Neg+(e_Neg'*N_Ker_N*e_Neg)*e_Pos/(N_Neg^2));   % p_i�������븺����֮��ľ���
   f_d = max(P_N);
   
   N_N = sqrt(diag(N_Ker_N)-2*N_Ker_N*e_Neg/N_Neg+(e_Neg'*N_Ker_N*e_Neg)*e_Neg/(N_Neg^2));   % n_i�������븺����֮��ľ���
   s_s = max(N_N);
   delta_Neg = 0.1*s_s;
   N_P = sqrt(diag(N_Ker_N)-2*P_Ker_N'*e_Pos/N_Pos+(e_Pos'*P_Ker_P*e_Pos)*e_Neg/(N_Pos^2));  % n_i��������������֮��ľ���
   s_d = max(N_P);   
   
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
%    s(Label==1) = s_Pos; %������ģ��������
%    s(Label==-1) = s_Neg; %������ģ��������
   
   
   Mem1=e_Pos-P_P/(f_s+10e-7);%������
   Nmem1=e_Pos-P_N/(f_d+10e-7);%��������

    Mem2=e_Neg-N_N/(s_s+10e-7);%������
    Nmem2=e_Neg-N_P/(s_d+10e-7);%��������
    
    s1=sqrt((Mem1.^2+(e_Pos-Nmem1).^2)./2);
    s2=sqrt((Mem2.^2+(e_Neg-Nmem2).^2)./2);
    
%     ��u������߼�����
%     s1(P_P>=P_N) = u*s1(P_P>=P_N);
%     s1(P_P<P_N) = 1*s1(P_P<P_N);
% 
%     s2(N_N>=N_P) = u*s2(N_N>=N_P);
%     s2(N_N<N_P) = 1*s2(N_N<N_P);

   s.s1 = s1; %������ģ��������
   s.s2 = s2; %������ģ��������
   
   
   
   
   



end

