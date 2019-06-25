function [train_err,test_err,train_time,test_time,Testing_time1,Training_time1] = bls_train_input(train_x,train_y,train_xf,train_yf,test_x,test_y,s,C,N1,N2,N3,m,l)

%Incremental Learning Process of the proposed broad learning system: for
%increment of input patterns
%Input: 
%---train_x,test_x : the training data and learning data in the begining of
%the incremental learning
%---train_y,test_y : the label
%---train_yf,train_xf: the whold training samples of the learning system
%---We: the randomly generated coefficients of feature nodes
%---wh:the randomly generated coefficients of enhancement nodes
%----s: the shrinkage parameter for enhancement nodes
%----C: the regularization parameter for sparse regualarization
%----N1: the number of feature nodes  per window
%----N2: the number of windows of feature nodes
%----N3: the number of enhancements nodes
% ---m:number of added input patterns per increment step
% ------l: steps of incremental learning

%output:
%---------Testing_time1:Accumulative Testing Times
%---------Training_time1:Accumulative Training Time

N11=N1;train_err=zeros(1,l);test_err=zeros(1,l);train_time=zeros(1,l);test_time=zeros(1,l);l2=zeros(1,l);
%%%%%%%%%%%%%%feature nodes%%%%%%%%%%%%%%
tic
train_x = zscore(train_x')';
H1 = [train_x .1 * ones(size(train_x,1),1)];y=zeros(size(train_x,1),N2*N11);
for i=1:N2
    we=2*rand(size(train_x,2)+1,N1)-1;
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
    y(:,N11*(i-1)+1:N11*i)=T1;
end
%%%%%%%%%%%%%enhancement nodes%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear T1;
H2 = [y .1 * ones(size(y,1),1)];

if N1*N2>=N3
     wh=orth(2*rand(N2*N1+1,N3)-1);
else
    wh=orth(2*rand(N2*N1+1,N3)'-1)'; 
end
Wh{1}=wh;
T2 = H2 * wh;
l2(1) = max(max(T2));
l2(1) = s/l2(1);
fprintf(1,'Enhancement nodes: Max Val of Output %f Min Val %f\n',l2(1),min(T2(:)));

T2 = tansig(T2 * l2(1));
T3=[y T2];
clear T2;
beta = (T3'  *  T3+eye(size(T3',1)) * (C)) \ ( T3' );
beta2=beta*train_y;

Training_time=toc;train_time(1,1) =Training_time;


disp('Training has been finished!');
disp(['The Total Training Time is : ', num2str(Training_time), ' seconds' ]);
xx = T3 * beta2;

%%%%%%%%%%%%%%%%%Training Accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%
yy = result(xx);
train_yy = result(train_y);
TrainingAccuracy = length(find(yy == train_yy))/size(train_yy,1);
disp(['Training Accuracy is : ', num2str(TrainingAccuracy * 100), ' %' ]);
train_err(1,1)=TrainingAccuracy;

%%%%%%%%%%%%%%%%%%%%%%Testing Process%%%%%%%%%%%%%%%%%%%
tic;
test_x = zscore(test_x')';
HH1 = [test_x .1 * ones(size(test_x,1),1)];
yy1=zeros(size(test_x,1),N2*N11);
for i=1:N2
    beta1=beta11{i};ps1=ps(i);
    TT1 = HH1 * beta1;
    TT1  =  mapminmax('apply',TT1',ps1)';

    clear beta1; clear ps1;
    %yy1=[yy1 TT1];
    yy1(:,N11*(i-1)+1:N11*i)=TT1;
end
clear TT1;
HH2 = [yy1 .1 * ones(size(yy1,1),1)]; 
TT2 = tansig(HH2 * wh * l2(1));TT3=[yy1 TT2];
clear wh;clear TT2;
%%%%%%%%%%%%%%%%% testing accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = TT3 * beta2;
y1 = result(x);
test_yy = result(test_y);
TestingAccuracy = length(find(y1 == test_yy))/size(test_yy,1);
Testing_time=toc;test_time(1,1) = Testing_time;test_err(1,1)=TestingAccuracy;
disp('Testing has been finished!');
disp(['The Total Testing Time is : ', num2str(Testing_time), ' seconds' ]);
disp(['Testing Accuracy is : ', num2str(TestingAccuracy * 100), ' %' ]);
%%%%%%%%%%%%%incremental training steps%%%%%%%%%%%%%%%%%%%
for e=1:l-1
    tic
    train_xx= zscore(train_xf((10000+(e-1)*m+1):(10000+e*m),:)')';
    train_yx= train_yf((10000+(e-1)*m+1):(10000+e*m),:);
    train_y1=train_yf(1:10000+e*m,:);
    Hx1 = [train_xx .1 * ones(size(train_xx,1),1)];yx=[];
    for i=1:N2
        beta1=beta11{i};ps1=ps(i);
        Tx1 = Hx1 * beta1;
        Tx1  =  mapminmax('apply',Tx1',ps1)';
        % clear beta1; clear ps1;
        yx=[yx Tx1];
    end
    Hx2 = [yx .1 * ones(size(yx,1),1)];
    wh=Wh{1};
    t2=Hx2 * wh;
    %     l2(e+1) = max(max(t2));
    %     l2(e+1) = s/l2(e+1);
    fprintf(1,'Enhancement nodes in incremental setp %f: Max Val of Output %f Min Val %f\n',e,l2(1),min(t2(:)));
    % yx=(yx-1)*2+1;
    t2 = tansig(t2 * l2(1));
    t2=[yx t2];
    betat = (t2'  *  t2+eye(size(t2',1)) * (C)) \ ( t2' );
    beta=[beta betat];
    beta2=beta*train_y1;
    T3=[T3;t2];
    Training_time=toc;
    train_time(1,e+1) =Training_time;
    xx = T3 * beta2;
    yy = result(xx);
    train_yy = result(train_y1);
    TrainingAccuracy = length(find(yy == train_yy))/size(train_yy,1);
    train_err(1,e+1)=TrainingAccuracy;
    disp(['Training Accuracy is : ', num2str(TrainingAccuracy * 100), ' %' ]);
    %%%%%%%%%%%%%incremental testing steps%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic
    x = TT3 * beta2;
    y1 = result(x);
    test_yy = result(test_y);
    TestingAccuracy = length(find(y1 == test_yy))/size(test_yy,1);
    Testing_time=toc;test_time(1,e+1) = Testing_time;test_err(1,e+1)=TestingAccuracy;
    disp('Testing has been finished!');
    disp(['The Total Testing Time is : ', num2str(Testing_time), ' seconds' ]);
    disp(['Testing Accuracy is : ', num2str(TestingAccuracy * 100), ' %' ]);
end
temp_test=sum(test_time);temp_train=sum(train_time);
Testing_time1=temp_test;Training_time1=temp_train;
end
