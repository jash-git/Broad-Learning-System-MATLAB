function [TrainingAccuracy,TestingAccuracy,Training_time,Testing_time] = bls_train_bp(train_x,train_y,test_x,test_y,s,C,N1,N2,N3)
% Learning Process of the proposed broad learning system trained Wh by BP
% algorithms. Moreover, the number of epochs of the bp algorithm is set as
% 50 by default.

%Input: 
%---train_x,test_x : the training data and learning data 
%---train_y,test_y : the label 
%---We: the randomly generated coefficients of feature nodes
%---wh:the randomly generated coefficients of enhancement nodes
%----s: the shrinkage parameter for enhancement nodes
%----C: the regularization parameter for sparse regualarization
%----N11: the number of feature nodes  per window
%----N2: the number of windows of feature nodes

%%%%%%%%%%%%%%feature nodes%%%%%%%%%%%%%%
tic
train_x = zscore(train_x')';
H1 = [train_x .1 * ones(size(train_x,1),1)];y=zeros(size(train_x,1),N2*N1);
for i=1:N2
    we=2*rand(size(train_x,2)+1,N1)-1;
    We{i}=we;
    A1 = H1 * we;A1 = mapminmax(A1);
    clear we;
    beta1  =  sparse_bls(A1,H1,1e-3,50)';
    beta11{i}=beta1;
    % clear A1;

    T1 = H1 * beta1;
    fprintf(1,'Feature nodes in window %f: Max Val of Output %f Min Val %f\n',i,max(T1(:)),min(T1(:)));

    [T1,ps1]  =  mapminmax(T1',0,1);T1 = T1';
    ps(i)=ps1;
    % clear H1;
    % y=[y T1];
    y(:,N1*(i-1)+1:N1*i)=T1;
end

clear H1;
clear T1;
%%%%%%%%%%%%%enhancement nodes%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H2 = [y .1 * ones(size(y,1),1)];
if N1*N2>=N3
     wh=orth(2*rand(N2*N1+1,N3)-1);
else
    wh=orth(2*rand(N2*N1+1,N3)'-1)'; 
end
T2 = H2 *  wh;
l2 = max(max(T2));
l2 = s/l2;
fprintf(1,'Enhancement nodes: Max Val of Output %f Min Val %f\n',l2,min(T2(:)));

T2 = tansig(T2 * l2);
T3=[y T2];
% clear H2;clear T2;
beta = (T3'  *  T3+eye(size(T3',1)) * (C)) \ ( T3'  *  train_y);

%%%%%%%%%%%%%%%back propogation for wh%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xx = T3 * beta;
yy = result(xx);
train_yy = result(train_y);
TrainingAccuracy = length(find(yy == train_yy))/size(train_yy,1);
disp(['Training Accuracy is : ', num2str(TrainingAccuracy * 100), ' %' ]);
test_x = zscore(test_x')';
HH1 = [test_x .1 * ones(size(test_x,1),1)];
%clear test_x;
yy1=zeros(size(test_x,1),N2*N1);
for i=1:N2
    beta1=beta11{i};ps1=ps(i);
    TT1 = HH1 * beta1;
    TT1  =  mapminmax('apply',TT1',ps1)';

    clear beta1; clear ps1;
    %yy1=[yy1 TT1];
    yy1(:,N1*(i-1)+1:N1*i)=TT1;
end
HH2 = [yy1 .1 * ones(size(yy1,1),1)]; 
TT2 = tansig(HH2 * wh * l2);TT3=[yy1 TT2];
clear TT2;
x = TT3 * beta;
y1 = result(x);
test_yy = result(test_y);
TestingAccuracy = length(find(y1 == test_yy))/size(test_yy,1);
clear TT3;
disp(['Testing Accuracy is : ', num2str(TestingAccuracy * 100), ' %' ]);

m=1;N=size(y,1);wh1=ones(1,N);Accuracy_b=zeros(1,5);
beta_b=zeros(size(beta,1),size(beta,2),5);
T3_b=zeros(size(T3,1),size(T3,2),5);
wh_b=zeros(size(wh,1),size(wh,2),5);
l2_b=zeros(size(l2,1),size(l2,2),5);
T2_b=zeros(size(T2,1),size(T2,2),5);
while m<=50
    xx = T3 * beta;
    W2=beta(N1*N2+1:end,:);
    Wh=wh(1:N1*N2,:);
    Wh_beta=wh(N1*N2+1,:);
    %T2temp=(T2.*(1-T2));
    T2temp=((1-T2.^2));
    tempWh=y'*(((xx-train_y)*W2').*T2temp);
    Wh=Wh+0.1*tempWh;
    %Wh=Wh+0.1*l2*tempWh;
    tempWhbeta=wh1*(((xx-train_y)*W2').*T2temp);
    Wh_beta=Wh_beta+0.1*tempWhbeta;
    %Wh_beta=Wh_beta+0.1*l2*tempWhbeta;
    wh_temp=[Wh; Wh_beta];
    T2_temp = H2 * wh_temp;
    l2_temp = max(max(T2_temp));
    l2_temp = s/l2_temp;
    fprintf(1,'Enhancement nodes in epochs %f : Max Val of Output %f Min Val %f\n',m,l2_temp,min(T2_temp(:)));
    T2_temp = tansig(T2_temp * l2_temp);
    T3_temp=[y T2_temp];
    beta_temp = (T3_temp'  *  T3_temp+eye(size(T3_temp',1)) * (C)) \ ( T3_temp'  *  train_y);
    %%%%%%%%%%%%%%%%%%%%%%%%%%The following block is for the case that
    %%%%%%%%%%%%%%%%%%%%%%%%%%when the next 4 iterations' training accuracy
    %%%%%%%%%%%%%%%%%%%%%%%%%%is not better than the result of the mth
    %%%%%%%%%%%%%%%%%%%%%%%%%%iteration.%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    xx_temp = T3_temp * beta_temp;
    yy_temp = result(xx_temp);
    train_yy = result(train_y);
    TrainingAccuracy = length(find(yy_temp == train_yy))/size(train_yy,1);
    disp(['Training Accuracy is : ', num2str(TrainingAccuracy * 100), ' %' ]);
    Accuracy=TrainingAccuracy;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%The following block is for the case that
    %%%%%%%%%%%%%%%%%%%%%%%%%%when the next 4 iterations' testing accuracy
    %%%%%%%%%%%%%%%%%%%%%%%%%%is not better than the result of the mth
    %%%%%%%%%%%%%%%%%%%%%%%%%%iteration.%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     TT2 = tansig(HH2 * wh_temp * l2_temp);TT3=[yy1 TT2];
%     clear TT2;
%     x = TT3 * beta_temp;
%     y1 = result(x);
%     test_yy = result(test_y);
%     TestingAccuracy = length(find(y1 == test_yy))/size(test_yy,1);
%     clear TT3;
%     disp(['Testing Accuracy is : ', num2str(TestingAccuracy * 100), ' %' ]);
%    Accuracy=TestingAccuracy;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if m<=5      
        beta=beta_temp; beta_b(:,:,m)=beta_temp;
        T3=T3_temp;T3_b(:,:,m)=T3_temp;
        wh=wh_temp;wh_b(:,:,m)=wh_temp;
        l2=l2_temp;l2_b(:,:,m)=l2_temp;
        T2=T2_temp;T2_b(:,:,m)=T2_temp;
        Accuracy_b(m)=Accuracy;
        m=m+1;
    else
        beta_b(:,:,1:4)=beta_b(:,:,2:5); beta_b(:,:,5)=beta_temp;
        T3_b(:,:,1:4)=T3_b(:,:,2:5);T3_b(:,:,5)=T3_temp;
        wh_b(:,:,1:4)=wh_b(:,:,2:5);wh_b(:,:,5)=wh_temp;
        l2_b(:,:,1:4)=l2_b(:,:,2:5);l2_b(:,:,5)=l2_temp;
        T2_b(:,:,1:4)=T2_b(:,:,2:5);T2_b(:,:,5)=T2_temp;
        Accuracy_b(1:4)=Accuracy_b(2:5);Accuracy_b(5)=Accuracy;
         [a,index]=max(Accuracy_b);
        if index~=1
            beta=beta_temp;
            T3=T3_temp;
            wh=wh_temp;
            l2=l2_temp;
            T2=T2_temp;
            m=m+1;
        else
            beta=beta_b(:,:,1);
            T3=T3_b(:,:,1);
            wh=wh_b(:,:,1);
            l2=l2_b(:,:,1);
            T2=T2_b(:,:,1);
            break
        end 
    end
end

Training_time = toc;
disp('Training has been finished!');
disp(['The Total Training Time is : ', num2str(Training_time), ' seconds' ]);

%%%%%%%%%%%%%%%%%Training Accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%
xx = T3 * beta;
clear T3;
yy = result(xx);
train_yy = result(train_y);
TrainingAccuracy = length(find(yy == train_yy))/size(train_yy,1);
disp(['Training Accuracy is : ', num2str(TrainingAccuracy * 100), ' %' ]);
tic;
%%%%%%%%%%%%%%%%%%%%%%Testing Process%%%%%%%%%%%%%%%%%%%
% test_x = zscore(test_x')';
% HH1 = [test_x .1 * ones(size(test_x,1),1)];
% %clear test_x;
% yy1=zeros(size(test_x,1),N2*N1);
% for i=1:N2
%     beta1=beta11{i};ps1=ps(i);
%     TT1 = HH1 * beta1;
%     TT1  =  mapminmax('apply',TT1',ps1)';
% 
%     clear beta1; clear ps1;
%     %yy1=[yy1 TT1];
%     yy1(:,N1*(i-1)+1:N1*i)=TT1;
% end
% clear TT1;clear HH1;
% HH2 = [yy1 .1 * ones(size(yy1,1),1)]; 
TT2 = tansig(HH2 * wh * l2);TT3=[yy1 TT2];
clear HH2;clear wh;clear TT2;
%%%%%%%%%%%%%%%%% testing accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = TT3 * beta;
y = result(x);
test_yy = result(test_y);
TestingAccuracy = length(find(y == test_yy))/size(test_yy,1);
clear TT3;

Testing_time = toc;
disp('Testing has been finished!');
disp(['The Total Testing Time is : ', num2str(Testing_time), ' seconds' ]);
disp(['Testing Accuracy is : ', num2str(TestingAccuracy * 100), ' %' ]);
