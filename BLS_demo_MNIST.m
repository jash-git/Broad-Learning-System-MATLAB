%%%%%%%%%%%%%%%%%%%%%%%%This is the demo for the bls models including the
%%%%%%%%%%%%%%%%%%%%%%%%proposed incremental learning algorithms. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%load the dataset MNIST dataset%%%%%%%%%%%%%%%%%%%%
clear; 
warning off all;
format compact;
load mnist;

%%%%%%%%%%%%%%%the samples from the data are normalized and the lable data
%%%%%%%%%%%%%%%train_y and test_y are reset as N*C matrices%%%%%%%%%%%%%%
train_x = double(train_x/255);
train_y = double(train_y);
% test_x = double(train_x/255);
% test_y = double(train_y);
test_x = double(test_x/255);
test_y = double(test_y);
train_y=(train_y-1)*2+1;
test_y=(test_y-1)*2+1;
assert(isfloat(train_x), 'train_x must be a float');
assert(all(train_x(:)>=0) && all(train_x(:)<=1), 'all data in train_x must be in [0:1]');
assert(isfloat(test_x), 'test_x must be a float');
assert(all(test_x(:)>=0) && all(test_x(:)<=1), 'all data in test_x must be in [0:1]');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Press any key to run the one shot BLS demo');
pause

%%%%%%%%%%%%%%%%%%%%This is the model of broad learning sytem with%%%%%%
%%%%%%%%%%%%%%%%%%%%one shot structrue%%%%%%%%%%%%%%%%%%%%%%%%
C = 2^-30; s = .8;%the l2 regularization parameter and the shrinkage scale of the enhancement nodes
N11=10;%feature nodes  per window
N2=10;% number of windows of feature nodes
N33=11000;% number of enhancement nodes
epochs=10;% number of epochs 
train_err=zeros(1,epochs);test_err=zeros(1,epochs);
train_time=zeros(1,epochs);test_time=zeros(1,epochs);
% rand('state',67797325)   % 12000 %%%%% The random seed recommended by the
% reference HELM [10].
N1=N11; N3=N33;  
for j=1:epochs    
    [TrainingAccuracy,TestingAccuracy,Training_time,Testing_time] = bls_train(train_x,train_y,test_x,test_y,s,C,N1,N2,N3);       
    train_err(j)=TrainingAccuracy * 100;
    test_err(j)=TestingAccuracy * 100;
    train_time(j)=Training_time;
    test_time(j)=Testing_time;
end
save ( ['mnist_result_oneshot_' num2str(N3)], 'train_err', 'test_err', 'train_time', 'test_time');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Press any key to run the one shot BLS demo with BP algorithm');
pause

%%%%%%%%%%%%%%%%%%%%This is the model of broad learning system for one%%%%%%
%%%%%%%%%%%%%%%%%%%%shot structrue with fine tuning under BP algorithm%%%%%%%%%%%%%%%%%%%%%%%%
C = 2^-30; s = .8;%the l2 regularization parameter and the shrinkage scale of the enhancement nodes
N11=10;%feature nodes  per window
N2=10;% number of windows of feature nodes
N33=5000;% number of enhancement nodes
epochs=1;% number of epochs 
train_err=zeros(1,epochs);test_err=zeros(1,epochs);
train_time=zeros(1,epochs);test_time=zeros(1,epochs);
% rand('state',67797325)   % 12000 %%%%% The random seed recommended by the
% reference HELM [10].
N1=N11; N3=N33;  
for j=1:epochs   
    [TrainingAccuracy,TestingAccuracy,Training_time,Testing_time] = bls_train_bp(train_x,train_y,test_x,test_y,s,C,N1,N2,N3);       
    train_err(j)=TrainingAccuracy * 100;
    test_err(j)=TestingAccuracy * 100;
    train_time(j)=Training_time;
    test_time(j)=Testing_time;
end
save ( ['mnist_result_bp_' num2str(N3)], 'train_err', 'test_err', 'train_time', 'test_time');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Press any key to run the BLS demo for increment of m enhancement nodes');
pause

%%%%%%%%%%%%%%%%%%%%This is the model of broad learning system for%%%%%%
%%%%%%%%%%%%%%%%%%%%increment of m enhancement nodes%%%%%%%%%%%%%%%%%%%%%%%%
C = 2^-30; s = .8;%the l2 regularization parameter and the shrinkage scale of the enhancement nodes
N11=10;%feature nodes  per window
N2=10;% number of windows of feature nodes
N33=9000;% number of enhancement nodes
m=500;%number of enhancement nodes in each incremental learning 
l=5;% steps of incremental learning
epochs=1;% number of epochs 
% rand('state',67797325)   % 12000 %%%%% The random seed recommended by the
% reference HELM [10].
N1=N11; N3=N33;  
[train_err,test_err,train_time,test_time,Testing_time,Training_time] = bls_train_enhance(train_x,train_y,test_x,test_y,s,C,N1,N2,N3,epochs,m,l); 
save ( ['mnist_result_enhance'], 'train_err', 'test_err', 'train_time', 'test_time','Testing_time','Training_time');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Press any key to run the BLS demo for increment of m2+m3 enhancement nodes and m1 feature nodes');
pause

%%%%%%%%%%%%%%%%%%%%This is the model of broad learning system for%%%%%%
%%%%%%%%%%%%%%%%%%%%increment of m2+m3 enhancement nodes and m1 feature nodes %%%%%%%%%%%%%%%%%%%%%%%%
C = 2^-30;%the regularization parameter for sparse regualarization
s = .8;%the shrinkage parameter for enhancement nodes
N11=10;%feature nodes  per window
N2=6;% number of windows of feature nodes
N33=3000;% number of enhancement nodes
epochs=1;% number of epochs 
m1=10;%number of feature nodes per increment step
m2=750;%number of enhancement nodes related to the incremental feature nodes per increment step
m3=1250;%number of enhancement nodes in each incremental learning 
l=5;% steps of incremental learning
train_err_t=zeros(epochs,l);test_err_t=zeros(epochs,l);train_time_t=zeros(epochs,l);test_time_t=zeros(epochs,l);
Testing_time_t=zeros(epochs,1);Training_time_t=zeros(epochs,1);
% rand('state',67797325)   % 12000 %%%%% The random seed recommended by the
% reference HELM [10].
N1=N11; N3=N33;  
for i=1:epochs      
    [train_err,test_err,train_time,test_time,Testing_time,Training_time] = bls_train_enhancefeature(train_x,train_y,test_x,test_y,s,C,N1,N2,N3,m1,m2,m3,l); 
    train_err_t(i,:)=train_err;test_err_t(i,:)=test_err;train_time_t(i,:)=train_time;test_time_t(i,:)=test_time;
    Testing_time_t(i)=Testing_time;Training_time_t(i)=Training_time;
end

save ( [ 'mnist_result_enhancefeature'], 'train_err_t', 'test_err_t', 'train_time_t', 'test_time_t','Testing_time_t','Training_time_t');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Press any key to run the BLS demo for increment of  m input patterns');
pause

%%%%%%%%%%%%%%%%%%%%This is the model of broad learning system for%%%%%%
%%%%%%%%%%%%%%%%%%%% increment of m input patterns %%%%%%%%%%%%%%%%%%%%%%%%
train_xf=train_x;train_yf=train_y;
train_x=train_xf(1:10000,:);train_y=train_yf(1:10000,:); % the selected input patterns of int incremental learning
C = 2^-30;%the regularization parameter for sparse regualarization
s = .8;%the shrinkage parameter for enhancement nodes
N11=10;%feature nodes  per window
N2=10;% number of windows of feature nodes
N33=5000;% number of enhancement nodes
epochs=1;% number of epochs 
m=10000;%number of added input patterns per increment step
l=6;% steps of incremental learning
train_err_t=zeros(epochs,l);test_err_t=zeros(epochs,l);train_time_t=zeros(epochs,l);test_time_t=zeros(epochs,l);
Testing_time_t=zeros(epochs,1);Training_time_t=zeros(epochs,1);
% rand('state',67797325)   % 12000 %%%%% The random seed recommended by the
% reference HELM [10].
N1=N11; N3=N33;  
for i=1:epochs        
    [train_err,test_err,train_time,test_time,Testing_time,Training_time] = bls_train_input(train_x,train_y,train_xf,train_yf,test_x,test_y,s,C,N1,N2,N3,m,l); 
    train_err_t(i,:)=train_err;test_err_t(i,:)=test_err;train_time_t(i,:)=train_time;test_time_t(i,:)=test_time;
    Testing_time_t(i)=Testing_time;Training_time_t(i)=Training_time;
end
save ( [ 'mnist_result_input'], 'train_err_t', 'test_err_t', 'train_time_t', 'test_time_t','Testing_time_t','Training_time_t');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Press any key to run the BLS demo for increment of m input patterns and m2 enhancement nodes');
pause

%%%%%%%%%%%%%%%%%%%%This is the model of broad learning system for%%%%%%
%%%%%%%%%%%%%%%%%%%%the increment of m input patterns and m2 enhancement nodes %%%%%%%%%%%%%%%%%%%%%%%%
% train_xf=train_x;train_yf=train_y;
% train_x=train_xf(1:10000,:);train_y=train_yf(1:10000,:); % the selected input patterns of int incremental learning
C = 2^-30;%the regularization parameter for sparse regualarization
s = .8;%the shrinkage parameter for enhancement nodes
N11=10;%feature nodes  per window
N2=10;% number of windows of feature nodes
N33=3000;% number of enhancement nodes
epochs=1;% number of epochs 
m=10000;%number of added input patterns per incremental step
m2=1600; %number of added enhancement nodes per incremental step
l=6;% steps of incremental learning
train_err_t=zeros(epochs,l);test_err_t=zeros(epochs,l);train_time_t=zeros(epochs,l);test_time_t=zeros(epochs,l);
Testing_time_t=zeros(epochs,1);Training_time_t=zeros(epochs,1);
% rand('state',67797325)   % 12000 %%%%% The random seed recommended by the
% reference HELM [10].
N1=N11; N3=N33;  
for i=1:epochs        
    [train_err,test_err,train_time,test_time,Testing_time,Training_time] = bls_train_inputenhance(train_x,train_y,train_xf,train_yf,test_x,test_y,s,C,N1,N2,N3,m,m2,l); 
    train_err_t(i,:)=train_err;test_err_t(i,:)=test_err;train_time_t(i,:)=train_time;test_time_t(i,:)=test_time;
    Testing_time_t(i)=Testing_time;Training_time_t(i)=Training_time;
end
save ( [ 'mnist_result_inputenhance'], 'train_err_t', 'test_err_t', 'train_time_t', 'test_time_t','Testing_time_t','Training_time_t');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



