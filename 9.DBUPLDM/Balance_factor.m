function theta = Balance_factor(Data, Label, Kernel, u, p1)

% The code only fits for the binary classification problem  �ô�����ʺ϶���������

% Data: The classification data whose samples lie in row  ����λ�����еķ�������

% Label: The label of data



%% Main  
   N_Samples = length(Label);
   theta = zeros(N_Samples, 1);
 % Abstract the positive and negative data  ��ȡ��������
   Data_Pos = Data(Label==1, :); %��������
   N_Pos = sum(Label==1); %������������
   e_Pos = ones(N_Pos, 1);
   Data_Neg = Data(Label==-1, :); 
   N_Neg = sum(Label==-1);
   e_Neg = ones(N_Neg, 1);
%    theta_p = zeros(N_Pos, 1);
%    theta_n = zeros(N_Neg, 1);
% Generate theta
 for i = 1:N_Samples
     if Label(i)>0
   theta(i)=1+(((N_Neg/N_Pos)^(1/2))*(N_Neg-N_Pos))/(N_Neg+N_Pos); %������������Ӧƽ������
     else
   theta(i)=1+(((N_Pos/N_Neg)^(1/2))*(N_Pos-N_Neg))/(N_Neg+N_Pos); %������������Ӧƽ������
     end
 end

end

