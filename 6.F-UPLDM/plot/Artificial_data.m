%% Initilizing the enviroment
clear all
close all
clc
rand('state', 2022)
randn('state', 2022)
%% data
mul=[0.5,3]; % 均值
S1=[0.1 0;0 30]; % 协方差
data1=mvnrnd(mul, S1, 100); % 产生高斯分布数据
data1(:,3)=1;
% 第二组数据
mu2=[-0.5,3];
S2=[0.1 0;0 30];
data2=mvnrnd(mu2,S2,100);
data2(:,3)=-1;
% noises of p
mm1=0;
mu3=[0,0];
S3=[1 -0.8;-0.8 1];
data3=mvnrnd(mu3,S3,mm1);
data3(:,3)=-1;
% noises of n
mm2=mm1;
mu4=[0,0];
S4=[1 -0.8;-0.8 1];
data4=mvnrnd(mu4,S4,mm2);
data4(:,3)=1;
data_noise = [data3;data4];
% predict
mu5=[0.5,-2.8]; % 均值
S5=[0.2 0;0 2]; % 协方差
data5=mvnrnd(mu5, S5, 100); % 产生高斯分布数据
data5(:,3)=1;
% 第二组数据
mu55=[-0.5,3];
S55=[0.2 0;0 2];
data55=mvnrnd(mu55,S55,100);
data55(:,3)=-1;
%% all
data_train=[data1;data2;data_noise];
data_predict=[data5;data55];
% plot(data_noise(:,1),data_noise(:,2), 'go', 'MarkerSize',8, 'LineWidth', 2);

plot(data1(:,1),data1(:,2), 'r+', 'MarkerSize',8, 'LineWidth', 2);hold on;
plot(data2(:,1),data2(:,2), 'bx', 'MarkerSize',8, 'LineWidth', 2);
% plot(data5(:,1),data5(:,2), 'r.', 'MarkerSize',8, 'LineWidth', 2);
% plot(data55(:,1),data55(:,2), 'b*', 'MarkerSize',8, 'LineWidth', 2);
hold off;


%% Load and prepare the data
% Training data
Data_Train = data_train;
% Normalization
Data_Train = [mapminmax(Data_Train(:, 1:end-1)', 0, 1)', Data_Train(:, end)]; % Map the original data to value between [0, 1] by colum
Samples_Train = Data_Train(:, 1:end-1);
Labels_Train = Data_Train(:, end);

% Predicting data
Data_Predict = data_predict;
% Normalization
Data_Predict = [mapminmax(Data_Predict(:, 1:end-1)', 0, 1)', Data_Predict(:, end)]; % Map the original data to value between [0, 1] by colum
Samples_Predict = Data_Predict(:, 1:end-1);
Label_Predict  = Data_Predict (:, end);


%% Some public parameters
% F_LDM Type
FLDM_Type = 'F1_LDM';
Kernel.Type = 'RBF';

switch FLDM_Type
    case 'F1_LDM'
        if strcmp(Kernel.Type, 'Linear')
            u = 0.1;                  % OK
            lambda1 = 0.03125;        % OK
            lambda2 = 0.03125;        % OK
            C = 0.75;                  % OK
            
            Value_Contour = 1;
            Str_Legend = 'Linear F1\_LDM';
        elseif strcmp(Kernel.Type, 'RBF')
            u = 0.1;                % OK
            lambda1 = 0.015625;       % OK
            lambda2 = 0.015625;       % OK
            C = 100;                % OK
            Kernel.gamma = 10.9227;      % OK
            
            Value_Contour = 1;      % OK
            Str_Legend = 'RBF-kernel F1\_LDM';
        else
            disp('Wrong parameters are provided.')
            return
        end
    case 'F2_LDM'
        if strcmp(Kernel.Type, 'Linear')
            u = 0.1;             % OK
            lambda1 = 0.5;       % OK
            lambda2 = 0.0625;    % OK
            C = 100;             % OK
            
            Value_Contour = 1;
            Str_Legend = 'Linear F2\_LDM';
        elseif strcmp(Kernel.Type, 'RBF')
            u = 0.1;             %  OK    91.9%@u = 0.5; Chose from u_Interval = linspace(0.1, 0.5, 3);
            lambda1 = 0.0039063;    %  OK    91.9%@lambda1 = 2^(-8);      Chose from lambda1_Interval = 2.^(-8:-2);
            lambda2 = 0.0039063;    %  OK    91.9%@lambda2 = 2^(-8);      Chose from lambda2_Interval = 2.^(-8:-2);
            C = 1;              %  OK    91.9%@C = 2^2;               Chose from C_Interval = 2.^(-8:8);
            Kernel.gamma = 10.9068;    %  OK    91.9%@Kernel.gamma = 2^3;    Chose from gamma_Interval = 2.^(-4:4);
            
            Value_Contour =1e-2;      % Best@20150910   91.9%@Kernel.gamma = 2^3;
            Str_Legend = 'RBF-kernel F2\_LDM';
        else
            disp('Wrong parameters are provided.')
            return
        end
    otherwise
        fprintf('%g\s','  Wrong inputs are provided.');
        return
end
QPPs_Solver = 'QP_Matlab';


%% Train and predict
C_s.C = C*abs(Labels_Train);
tic
%C_s.s = Fuzzy_MemberShip(Samples_Train, Labels_Train,Kernel,u);
C_s.s = Fuzzy_MemberShip_FCM(Samples_Train, Labels_Train);
Outs_Train = Train_FLDM(Samples_Train, Labels_Train, lambda1, lambda2, C_s, FLDM_Type, Kernel, QPPs_Solver);
t = toc;
% Predict the data
[Acc, Margin, Data_Supporters, Label_Decision, Outs_Predict] = Predict_FLDM(Outs_Train, Samples_Predict, Label_Predict);


%% Statistical results
% Predicting accurate
disp(['  The training time is ', num2str(t), ' seconds.'])
disp(['  The predicting accurate is ', num2str(100*Acc), '%.'])
Margin_MEAN = Margin.MEAN;
Str_MEAN = sprintf('  The Margin MEAN is %0.2e', Margin_MEAN);
disp(Str_MEAN)
Margin_VARIANCE = Margin.VARIANCE;
Str_VARIANCE = sprintf('  The Margin VARIANCE is %0.2e', Margin_VARIANCE);
disp(Str_VARIANCE)

%% Visualization
%% hot map
xp = Samples_Train(Labels_Train==1,:);
xn = Samples_Train(Labels_Train==-1,:);
xp=[xp;xn];
figure
[xq,yq] = meshgrid(linspace(min(xp(:, 1)), max(xp(:, 1)), 100), linspace(min(xp(:, 2)), max(xp(:, 2)), 100));
vq = griddata(xp(:, 1), xp(:, 2),C_s.s,xq,yq,'v4');
% vq(find(vq<0))=0;
contourf(xq,yq,vq,100,'LineStyle','none')
colormap(jet);
colorbar;
hold on
plot(Samples_Train(Labels_Train==1, 1), Samples_Train(Labels_Train==1, 2), 'r+', 'MarkerSize',8, 'LineWidth', 2); % Positive trainging data
plot(Samples_Train(Labels_Train==-1, 1), Samples_Train(Labels_Train==-1, 2), 'bx', 'MarkerSize',8, 'LineWidth', 2); % Negative trainging data

%---
% The Intervals for both X and Y axise
x_Interval = linspace(min(Samples_Train(:, 1)), max(Samples_Train(:, 1)), 100);
y_Interval = linspace(min(Samples_Train(:, 2)), max(Samples_Train(:, 2)), 100);
% Contours
[X, Y, Z] = Contour_FLDM(Outs_Predict, x_Interval, y_Interval);
[Con_Decsi, h_Decsi] = contour(X, Y, Z, [0 0], '-', 'Color', 'k', 'LineWidth', 2);
clabel(Con_Decsi, h_Decsi, 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');


%---

plot(Samples_Train(201:end-mm2, 1), Samples_Train(201:end-mm2, 2), 'go', 'MarkerSize',8, 'LineWidth', 2); %noises of p
plot(Samples_Train(end-mm2+1:end, 1), Samples_Train(end-mm2+1:end, 2), 'gsquare', 'MarkerSize',8, 'LineWidth', 2); %noises of n
legend('Confidence Plane', 'Class 1','Class 2', 'Noises of Class1','Noises of Class2')
hold off;
figure(2)
plot(Samples_Train(Labels_Train==1, 1), Samples_Train(Labels_Train==1, 2), 'r+', 'MarkerSize',8, 'LineWidth', 2); % Positive trainging data
hold on;
plot(Samples_Train(Labels_Train==-1, 1), Samples_Train(Labels_Train==-1, 2), 'bx', 'MarkerSize',8, 'LineWidth', 2); % Negative trainging data
plot(Samples_Train(end-mm2, 1), Samples_Train(end-mm2, 2), 'go', 'MarkerSize',8, 'LineWidth', 2); %noises of p
plot(Samples_Train(end-mm2:end, 1), Samples_Train(end-mm2:end, 2), 'gsquare', 'MarkerSize',8, 'LineWidth', 2); %noises of n
plot(Data_Supporters(:, 1), Data_Supporters(:, 2), 'ko', 'MarkerSize',8, 'LineWidth', 2);  % The support vectors
legend('Class 1','Class 2', 'noises of Class1','noises of Class2' , 'Support vectors')
% The Intervals for both X and Y axise
x_Interval = linspace(min(Samples_Train(:, 1)), max(Samples_Train(:, 1)), 100);
y_Interval = linspace(min(Samples_Train(:, 2)), max(Samples_Train(:, 2)), 100);
% Contours
[X, Y, Z] = Contour_FLDM(Outs_Predict, x_Interval, y_Interval);
[Con_Pos, h_Pos] = contour(X, Y, Z, Value_Contour*[1 1], ':', 'Color', 'k', 'LineWidth', 1);

clabel(Con_Pos, h_Pos, 'Color','k', 'FontSize', 12, 'FontWeight', 'bold');
[Con_Decsi, h_Decsi] = contour(X, Y, Z, [0 0], '-', 'Color', 'k', 'LineWidth', 2);
clabel(Con_Decsi, h_Decsi, 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');
[Con_Neg, h_Neg] = contour(X, Y, Z, Value_Contour*[-1 -1], ':', 'Color','k', 'LineWidth', 1);
clabel(Con_Neg, h_Neg, 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');
clabel(Con_Neg, h_Neg, 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');