function [Train_x, Test_x]=pre_zca(train_x,test_x)

train_x = bsxfun(@rdivide, bsxfun(@minus, train_x, mean(train_x,2)), sqrt(var(train_x,[],2)+10));
 
C = cov(train_x);
M = mean(train_x);
[V,D] = eig(C);
P = V * diag(sqrt(1./(diag(D) + 1e2))) * V';
Train_x = bsxfun(@minus, train_x, M) * P;
 
test_x = bsxfun(@rdivide, bsxfun(@minus, test_x, mean(test_x,2)), sqrt(var(test_x,[],2)+10));
Test_x = bsxfun(@minus, test_x, M) * P;