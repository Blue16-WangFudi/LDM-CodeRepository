function [X, Y, Z] = Contour_FLDM(Outs_Predict, x_Interval, y_Interval)

%------------------- Input -------------------%
% Outs_Predict includes:  
%                Outs_Predict.u    The solution to the original problem    
%    Outs_Predict.Samples_Train    the samples
%           Outs_Predict.Kernel    kernel type

%                    x_Interval    The range of x-axis
%                    y_Interval    The range of y-axis

%------------------- Output -------------------%
% X, Y, Z for contours
% Author: Wendong Wang
%  Email: d.sylan@foxmail.com
%   data: 2015,9,8
% updata: 2015,10,7


 rand('state', 2015)
 randn('state', 2015)

%% Main
   u = Outs_Predict.u;
   b = u(end);
   Samples_Train = Outs_Predict.Samples_Train;
   Kernel = Outs_Predict.Kernel;

 % Contour
   [X, Y] = meshgrid(x_Interval, y_Interval);
   [m, n] = size(X);
   Data_Contour = [X(:), Y(:)];
   
 % Predict the label
   if strcmp(Kernel.Type, 'Linear')
       w = u(1:end-1);
       Value_Decision = Data_Contour*w + b*ones(m*n, 1);
   else
       alpha = u(1:end-1);
       Value_Decision = Function_Kernel(Data_Contour, Samples_Train, Kernel)*alpha + b*ones(m*n, 1);
   end
 % Compute the Z
   Z = reshape(Value_Decision, m, n);   
   
end

