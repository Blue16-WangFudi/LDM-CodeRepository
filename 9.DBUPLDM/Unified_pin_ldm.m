function [acc_f, C_f] = Unified_pin_ldm(X_train, Y_train, X_test,Y_test, kernel, tau, c1val,p1,lamb1, lamb2)
[m, n] = size(X_train);
H = zeros(m,m);
m1=size(X_test,1);
e = ones(m, 1);
if (kernel == 1)
    K = [X_train, e]';
    E = eye(n+1);
else
    K = [Function_Kernel(X_train, X_train,'RBF',p1), e]';
    E = blkdiag(Function_Kernel(X_train, X_train, 'RBF',p1), 1);
end

CR = 1e-7;
lambda2_eORs = lamb2*e;
K_IorS = K;

Q = E + 4*lamb1*(m*K_IorS*K_IorS'-K_IorS*Y_train*Y_train'*K_IorS')/(m^2);
if (kernel == 1)
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
acc=zeros(length(c1val),1);
for i = 1:length(c1val)
        C0= c1val(i);
    for ii=1:size(Y_train,1)
        if (Y_train(ii)==1)
            C(ii,:)= C0;
        else
            C(ii,:)= C0*(length(find(Y_train==-1)))/(length(find(Y_train==1)));
        end
    end
    
C_eORs = C;
mu0= zeros(m, 1);
Options.LargeScale = 'off';
Options.Display = 'off';
% Options.Algorithm = 'interior-point-convex';
% solver
H = KY'*inv(Q)*KY;
H = (H+H')/2;
z = H*lambda2_eORs/m-e;
Aeq = Y_train';
beq = 0;
lb = sign(-tau)*C_eORs.*abs(tau);
ub = C_eORs;
mu = quadprog(H, z, [], [], [], [], lb, ub, mu0, Options);
u = Q\KY*(lambda2_eORs/m+mu);
b = u(end);
omega = u(1:end-1);

%% predict
% H_test = zeros(m1, m);
% if(kernel==1)
%     for i=1:m1
%         for j=1:m
%             H_test(i,j) = svkernel('linear',X_test(i,:), X_train(j,:),p1);
%         end
%     end
% end
%
% if(kernel==2)
%     for i=1:m1
%         for j=1:m
%             H_test(i,j) = svkernel('rbf',X_test(i,:), X_train(j,:),p1);
%         end
%     end
% end
% pred_label = sign(H_test*(omega.* Y_train) +b);


pred_label = -ones(m1, length(C));
if (kernel == 1)
    Value_Decision = X_test*omega + b*ones(m1,1);
else
    Value_Decision = Function_Kernel(X_test, X_train, 'RBF',p1)*omega + b*ones(m1,1);
end
pred_label(Value_Decision>=0,i) = 1;
acc(i)= length(find(pred_label(:,i)==Y_test))*100/length(Y_test);
end
[acc_f,temp] = max(acc);
C_f = c1val(temp);
end
