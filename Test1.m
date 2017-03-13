%%Experiment 1: Simulated Data parfor small p   (n>p)
tic;
clear all;

n = 100; N1 = 70; N2 = 30;
P = 200; P1 = 120; P2 = 80;

X1 = randn(N1,P1);
X2 = randn(N2,P1) + 1.5;
X3 = randn(n,P2);
X = [[X1;X2],X3];


K = 2;
lambda = 8;
penatype = 'l1';

%Cross Validation
Fold_Number = 5;
indices = crossvalind('Kfold',n,Fold_Number);
s = 0;
Ind = [];
for i = 1:Fold_Number
    validate = (indices == i); train = ~validate;
     X_train = X(train,:); X_validate = X(validate,:);
     [Mu,Pi,Sigma,CluResult,INDEX,LogLF] = EMforMPLE(X(train,:),X(validate,:),K,lambda,penatype);
     s = s + LogLF;
     Ind = union(Ind,INDEX);
end
MeanLogLF = s/Fold_Number
Ind
toc;