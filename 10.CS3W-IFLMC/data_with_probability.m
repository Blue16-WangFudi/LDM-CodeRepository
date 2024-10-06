function [proba,A,B,fval] = data_with_probability(F,Labels_Predict,x0)



% syms x1 x2
n_positive = length(find(Labels_Predict==1));
t_i = zeros(length(Labels_Predict),1);
for i = 1:length(Labels_Predict)
    if Labels_Predict(i)==1
        t_i(i)=(n_positive+1)/(n_positive+2);
    else
        t_i(i)=1/(n_positive+2);
    end
end
e = ones(length(Labels_Predict),1);
% P = e./(1+exp(x1*F+x2));
func =@(x) -sum(t_i.*log(e./(1+exp(x(1)*F+x(2))))+(1-t_i).*log(1-(e./(1+exp(x(1)*F+x(2))))));
% [n,~,list_x1,list_x2] = fastest_gradient_descent_update(func,x10,x20,error);
% [n,list_x1,list_x2] = Newton_two_dimension(func,x10,x20,error)

% A = list_x1(end);
% B = list_x2(end);

[x_matrix,fval] = fminunc(func,x0);
A = x_matrix(1);
B = x_matrix(2);
proba = e./(1+exp(A*F+B));
end

