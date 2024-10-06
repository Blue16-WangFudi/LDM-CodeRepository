function result  =  CS3WD(test_label, prob, cost_matrix)
% input
% test_label:  size: n*1
% prob: size: n*2
% cost_matrix: size: 2*3 row: 1-P 2-N, column: 1-P 3-N 2-B

bayes_loss = prob*cost_matrix;% the expected loss of test samples, bayes loss
[loss_sort, num_sort]=sort(bayes_loss,2,'ascend');%代价最小的决策
dec_CS3WD = num_sort(:,1);% size: n*1
    assert(test_num == size(dec_CS3WD,1),'decision CS3WD ?')

dec_res_CS = dec_CS3WD == test_label;% 正确判断的结果
err_CS3WD = 1 - sum(dec_res_CS)/test_num;%错误率

cost_i_CS = zeros(size(dec_CS3WD));
for i=1:test_num
    cost_i_CS(i) = cost_matrix(test_label(i),dec_CS3WD(i));
end
cost_CS3WD = sum(cost_i_CS);%总代价

% output
result.dec = dec_CS3WD;
result.cost = cost_CS3WD;
result.err = err_CS3WD;
result.bayes_risk = bayes_loss;