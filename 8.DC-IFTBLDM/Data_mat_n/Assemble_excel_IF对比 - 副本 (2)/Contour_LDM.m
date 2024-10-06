function [X, Y, Z] = Contour_LDM(Outs_Predict, x_Interval, y_Interval)

%------------------- Input -------------------%
% Outs_Predict includes:  
%    Outs_Predict.Samples_Train    the samples
%            Outs_Predict.alpha    The solution to the original problem    
%           Outs_Predict.Kernel    kernel type

%                    x_Interval    The range of x-axis
%                    y_Interval    The range of y-axis

%------------------- Output -------------------%
% X, Y, Z for contours


   rand('state', 2015)
   randn('state', 2015)


%% Main
   Samples_Train = Outs_Predict.Samples_Train;
   alpha = Outs_Predict.alpha;
   Kernel = Outs_Predict.Kernel;

 % Contour
   [X, Y] = meshgrid(x_Interval, y_Interval);
   [m, n] = size(X);
   Data_Contour = [X(:), Y(:)];
 % Predict the label
   Value_Decision = Function_Kernel(Data_Contour, Samples_Train, Kernel)*alpha; 
 % Compute the Z
   Z = reshape(Value_Decision, m, n);   
   
end

