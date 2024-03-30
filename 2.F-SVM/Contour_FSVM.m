function [X, Y, Z] = Contour_FSVM(Inputs, x_Interval, y_Interval)

% Inputs: It contains four parts: The training data, Traing data's label,
  % Kernel, Tradeoff ofr slack variables
  
% x_Interval: The range of x-axis

% y_Interval: The range of y-axis

% tau: To find the support vectors


   rand('state', 2015)
   randn('state', 2015)
   
   
%% Main
   Data_Train = Inputs.Data_Train;
   Samples_Train = Data_Train(:, 1:end-1);
   Labels_Train = Data_Train(:, end);
   Kernel = Inputs.Kernel;
   beta = Inputs.beta;
   b = Inputs.b;
   
   [X, Y] = meshgrid(x_Interval, y_Interval);
   [m, n] = size(X);
   Samples_Contour = [X(:), Y(:)];
   if strcmp(Kernel.Type, 'Linear')
       Value_Decision =  Samples_Contour*Samples_Train'*diag(Labels_Train)*beta+b*ones(m*n, 1);
   else
       Value_Decision =  Function_Kernel(Samples_Contour, Samples_Train, Kernel)*diag(Labels_Train)*beta+b*ones(m*n, 1);
   end
 % Compute the Z
   Z = reshape(Value_Decision, m, n);


end

