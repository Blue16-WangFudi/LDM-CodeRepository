function [Best_mean,Best_variance,acc_f, C_f] = Unified_pin_svm(X_train, Y_train, X_test,Y_test, kernel, tau, c1val,p1)
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
Mean=zeros(length(c1val),1);
Variance=zeros(length(c1val),1);
for i = 1:length(c1val)
    C0= c1val(i);
    for ii=1:size(Y_train,1)
        if (Y_train(ii)==1)
            C(ii,:)= C0;
        else
            C(ii,:)= C0*(length(find(Y_train==-1)))/(length(find(Y_train==1)));
        end
    end
    if(tau==0)
        H4 = H;
        f = -ones(m,1);
        Aeq = Y_train';
        beq= 0;
        LB = zeros(m,1);
        UB= C;
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
        beq= [0; C];
        LB = zeros(2*m,1);
        %%
        % options = optimset('Algorithm', 'Trust-region-reflective');
        % options.Display = 'off';
        % options.MaxIter = 500;
        options.LargeScale = 'off';
        options.Display = 'off';
        options.Algorithm = 'interior-point-convex';
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
    
    %%均值方差
    % 假设你已经有 α 和 b
    alpha = alpha_beta; % 用你的 α 替换

    % 通过找到非零 α_i 找到支持向量索引
    support_vector_indices = find(alpha > 0);

    % 初始化 w
    w = zeros(1, size(X_train, 2)); % 假设你有训练数据 X

    % 计算 w
    for ith = 1:length(support_vector_indices)
        idx = support_vector_indices(ith);
        w = w + alpha(idx) * Y_train(idx) * X_train(idx, :);
    end

    % 计算距离
    distances = abs(X_train * w' + b) / norm(w);

    % 计算距离均值和方差
    mean_distance = mean(distances);
    variance_distance = var(distances);
  
    Mean(i) = mean_distance;
    Variance(i) = variance_distance;
end

[acc_f,temp] = max(acc);
C_f = c1val(temp);
Best_mean = Mean(temp);
Best_variance = Variance(temp);
% spars = length((alpha_beta.* Y_train))- nnz(alpha_beta.* Y_train);
end
