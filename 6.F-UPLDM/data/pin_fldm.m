function [pred_label, alpha_beta,b] = pin_fldm(X_train, Y_train, X_test, kernel, tau, C,p1,s,lamb1, lamb2)
m = size(X_train,1);
H = zeros(m,m);
m1=size(X_test,1);
e = ones(m, 1);
if strcmp(kernel,1)
    K = [X_train, e]';
    E = eye(n+1);
else
    K = [svkernel('rbf',X_train, X_train,p1), e]';
    E = blkdiag(svkernel('rbf',X_train, X_train,p1), 1);
end

CR = 1e-7;
lambda2_eORs = lamb2*e;
C_eORs = C.*s;
K_IorS = K;

Q = E + 4*lamb1*(m*K_IorS*K_IorS'-K_IorS*Y_train*Y_train'*K_IorS')/(m^2);
if strcmp(kernel, 1)
    Q = Q + CR*eye(n+1);
else
    Q = Q + CR*eye(m+1);
end
KY = K*diag(Y_train);


% Add small amount of zero order regularisation to avoid problems
% when Hessian is badly conditioned.
% H = H+1e-10*eye(size(H));

%% Solving QPP
% Parameters for quadprog
mu0= zeros(m, 1);
Options.LargeScale = 'off';
Options.Display = 'off';
Options.Algorithm = 'interior-point-convex';
% solver
H = KY'*inv(Q)*KY;
H = (H+H')/2;
z = H*lambda2_eORs/m-e;
Aeq = Y_train';
beq = 0;
lb = -C.*abs(tau);
ub = C_eORs;
mu = quadprog(H, z, [], [], Aeq, beq, lb, ub, mu0, Options);
u = Q\KY*(lambda2_eORs/m+mu);
b = u(end);
alpha = u(1:end-1);

%% predict
H_test = zeros(m1, m);
if(kernel==1)
    for i=1:m1
        for j=1:m
            H_test(i,j) = svkernel('linear',X_test(i,:), X_train(j,:),p1);
        end
    end
end

if(kernel==2)
    for i=1:m1
        for j=1:m
            H_test(i,j) = svkernel('rbf',X_test(i,:), X_train(j,:),p1);
        end
    end
end
pred_label = sign(H_test*(mu.* Y_train) +b);

% 
% Label_Decision = -ones(m1, 1);
% if (kernel == 1)
%     for i = 1:m1
%         for j = 1:m
%             Value_Decision(i,j) = svkernel(X_test(i,:)*u + b*ones(m1,1);
%     
% else
%     Value_Decision = svkernel('rbf',X_test, X_train,p1)*alpha + b*ones(m1,1);
% end
% Label_Decision(Value_Decision>=0) = 1;
% pred_label = Label_Decision;
alpha_beta = mu;
% spars = length((alpha_beta.* Y_train))- nnz(alpha_beta.* Y_train);
end
