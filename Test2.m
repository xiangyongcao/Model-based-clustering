%%Experiment 2: validate modified BIC for l1 and nonconvex penalty
tic;
clear all

num = 40;    % the number of simulated data

K = [1,2,3];
lambda = 1:15;
lenK = length(K);
lenlam = length(lambda);

R = zeros(3,2,num);
S = zeros(num,3);
U = zeros(num,1);
V= zeros(num,1);

penatype = 'l1';

n = 20; N1 = 16; N2 = 4;
P = 200; P1 = 40; P2 = 160;


for i = 1:num
    i
    X1 = randn(N1,P1);
    X2 = randn(N2,P1) + 1.5;
    X3 = randn(n,P2);
    X = [[X1;X2],X3];
    
    
    
    R1 = zeros(3,lenlam);
    R2 = zeros(3,2);
    Fold_Number = 5;
    
    for j = 1:lenK
        for k = 1:lenlam
            indices = crossvalind('Kfold',n,Fold_Number);
            s = 0;
            for l = 1:Fold_Number
                validate = (indices == l); train = ~validate;
                X_train = X(train,:); X_validate = X(validate,:);
                [Mu,Pi,Sigma,CluResult,INDEX,LogLF] = EMforMPLE(X_train,X_validate,K(j),lambda(k),penatype);
                s = s + LogLF;
            end
            MeanLogLF = s/Fold_Number;
            R1(j,k) = MeanLogLF;
        end
        [m_value,m_index] = min(R1(j,:));
        S(i,j) = m_value;
        R2(j,:) = [j,lambda(m_index)];
    end
    
    R(:,:,i) = R2;
end

%table 1
MeanMeanLogLF = (sum(S)/num)';
StdMeanLogLF = [std(S(:,1));std(S(:,2));std(S(:,3))];

B = R(:,2,:);
C = B(:,:)';
Meanlambda = (sum(C)/num)';
Stdlambda = [std(C(:,1));std(C(:,2));std(C(:,3))];

time = toc;

save l1.mat time MeanMeanLogLF StdMeanLogLF Meanlambda Stdlambda;
clear;
load l1.mat;