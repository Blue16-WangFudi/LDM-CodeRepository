function [acc_f,C_f] = Unified_pin_fsvm(X_train, Y_train, X_test,Y_test, kernel, tau, c1val,p1,s)
m = size(X_train,1);
H = zeros(m,m);
m1=size(X_test,1);
%% Kernel Construction
if(kernel==1)
    for i=1:m
        for j=1:m
            H(i,j) = Y_train(i)*Y_train(j)*svkernel('linear',X_train(i,:), X_train(j,:),p1);
        end
    end
end

if(kernel==2)
    for i=1:m
        for j=1:m
            H(i,j) = Y_train(i)*Y_train(j)*svkernel('rbf',X_train(i,:), X_train(j,:),p1);
        end
    end
end
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
    C_s = C.*s;
    if(tau==0)
        H4 = H;
        f = -ones(m,1);
        Aeq = Y_train';
        beq= [0];
        LB = zeros(m,1);
        UB= C_s;
        options.Display = 'off';
        options.MaxIter = 500;
        alpha_beta = quadprog(H4, f, [], [], Aeq, beq, LB, UB, [],options);
        idx = find( (alpha_beta  > 1e-9) & ( alpha_beta  < (C-1e-19) ));
        if isempty(idx)
            b=0;
        else
            b=mean(Y_train(idx,1)-(H(idx,:)*(alpha_beta.*Y_train)));
        end
    else
        
        
        % Add small amount of zero order regularisation to avoid problems
        % when Hessian is badly conditioned.
        % H = H+1e-10*eye(size(H));
        
        %% Solving QPP given in eq 7
        H4 = [H, -sign(tau)*H; -sign(tau)*H, H];
        f = -[ones(m,1); -sign(tau)*ones(m,1)];
        Aeq = [Y_train', -sign(tau)*Y_train'; eye(m,m), sign(tau)*(1/tau)*eye(m,m)];
        beq= [0; C_s];
        LB = zeros(2*m,1);
        %%
        % options = optimset('Algorithm', 'Trust-region-reflective');
        options.Display = 'off';
        options.MaxIter = 500;
        lambda_beta = quadprog(H4, f, [], [], Aeq, beq, LB, [], [],options);
        alpha=lambda_beta(1:m,:);
        beta=lambda_beta(m+1:end,:);
        alpha_beta=alpha-sign(tau)*beta;
        
        %% For calculation of bias term (from eq 1 on page 4)
        idx = find( (abs(alpha )> 1e-9) & (abs(beta) > 1e-9));
        if isempty(idx)
            b=0;
        else
            b=mean(Y_train(idx,1)-(H(idx,:)*(alpha_beta.*Y_train)));
        end
        
        %%
    end
    H_test = zeros(m1, m);
    if(kernel==1)
        for ii=1:m1
            for j=1:m
                H_test(ii,j) = svkernel('linear',X_test(ii,:), X_train(j,:),p1);
            end
        end
    end
    
    if(kernel==2)
        for ii=1:m1
            for j=1:m
                H_test(ii,j) = svkernel('rbf',X_test(ii,:), X_train(j,:),p1);
            end
        end
    end
    pred_label = sign(H_test*(alpha_beta.* Y_train) +b);
    acc(i)= length(find(pred_label==Y_test))*100/length(Y_test);
end
[acc_f,temp] = max(acc);
C_f = c1val(temp);
% spars = length((alpha_beta.* Y_train))- nnz(alpha_beta.* Y_train);
end
