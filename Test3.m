%%Experiment 2: Frequencies of the selected numbers of clusters for l1 and nonconvex penalty
tic;
clear all

num = 40;

K = [2,3];
lambda = 1:10;
lenK = length(K);
lenlam = length(lambda);

n = 10; N1 = 3; N2 = 7;
P = 100; P1 = 50; P2 = 50;


S = zeros(3,num);
T = zeros(3,num);
IND3 = zeros(3,P,num);
Clu3 = zeros(3,n,num);

penatype = 'l1';



for i = 1:num
    i
    X1 = randn(N1,P1);
    X2 = randn(N2,P1) + 1.5;
    X3 = randn(n,P2);
    X = [[X1;X2],X3];
    
    R = zeros(3,lenlam);
    IND2 = [];
    Clu2 = [];
    Fold_Number = 5;
    
    for j = 1:lenK
        IND1 = [];
        Clu1 = [];
        for k = 1:lenlam
            indices = crossvalind('Kfold',n,Fold_Number);
            s = 0;
            Ind = [];
            for l = 1:Fold_Number
                validate = (indices == l); train = ~validate;
                X_train = X(train,:); X_validate = X(validate,:);
                [Mu,Pi,Sigma,CluResult,INDEX,LogLF] = EMforMPLE(X_train,X_validate,K(j),lambda(k),penatype);
                s = s + LogLF;
                Ind = union(Ind,INDEX);
            end
            MeanLogLF = s/Fold_Number;
            R(j,k) = MeanLogLF;
            
            len1 = P - length(Ind);
            APP = zeros(1,len1);
            Ind1 = [Ind,APP];
            
            IND1 = [IND1;Ind1];
            Temp = CluResult(1);
            if Temp == 2 || Temp == 3
                CluResult(CluResult == 2) = 10;
                CluResult(CluResult == 1) = 20;
            end
            
            Clu1 = [Clu1;CluResult];
        end
        [mValue,index] = min(R(j,:));
        
        S(j,i) = mValue;
        T(j,i) = lambda(index);
        
        temp = IND1(index,:);
        IND2 = [IND2;temp];
        
        temp11 = Clu1(index,:);
        Clu2 = [Clu2;temp11];
    end
    IND3(:,:,i) = IND2;
    Clu3(:,:,i) = Clu2;
end

[p,q] = min(S);  % p - the mimimum MeanLogLF of every simulated data
% q - the corresponding cluster label of the minimum MeanLogLF


Lambda = [];
for l = 1:num
    temp = T(:,l);
    Lambda = [Lambda,temp(q(l))];
end

IND = [];    % 100 x P
for l = 1:num
    temp = IND3(:,:,l);
    IND = [IND;temp(q(l),:)];
end

Clu = [];    % 100 x n
for l = 1:num
    temp = Clu3(:,:,l);
    Clu = [Clu;temp(q(l),:)];
end


%table 2

[m1 n1] = find(q==1);
[m2 n2] = find(q==2);
[m3 n3] = find(q==3);
fre1 = length(m1);
fre2 = length(m2);
fre3 = length(m3);

Fre = [fre1;fre2;fre3];        % Frequencies of the selected numbers of clusters

MeanLogLF1 = p(n1);
MeanLogLF2 = p(n2);
MeanLogLF3 = p(n3);
MeanMeanLogLF = [sum(MeanLogLF1)/fre1;sum(MeanLogLF2)/fre2;sum(MeanLogLF3)/fre3];
StdMeanLogLF = [std(MeanLogLF1);std(MeanLogLF2);std(MeanLogLF3)];  % corresponding mean and standard derivation of MeanLogLF

Lambda1 = Lambda(n1);
Lambda2 = Lambda(n2);
Lambda3 = Lambda(n3);
Meanlambda = [sum(Lambda1)/fre1;sum(Lambda2)/fre2;sum(Lambda3)/fre3];
Stdlambda = [std(Lambda1);std(Lambda2);std(Lambda3)];    % corresponding mean and standard derivation of lambda

%the number of NonZero excluded and the number of Zeros included(PZC)
IND1 = IND(n1,:);
IND2 = IND(n2,:);
IND3 = IND(n3,:);

F_index = 1:P;
s1 = 0;
t1 = 0;
for i = 1:fre1
    temp = IND1(i,:);
    zeros_in = length(intersect(F_index(P1+1:P),temp));
    s1 = s1 + zeros_in;
    
    temp1 = setdiff(F_index,temp);
    nonzeros_ex = length(intersect(F_index(1:P1),temp1));
    t1 = t1 + nonzeros_ex;
end
Mean1Zeros_In = s1/fre1;
Mean1NonZeros_Ex = t1/fre1;

s2 = 0;
t2 = 0;
for i = 1:fre2
    temp = IND2(i,:);
    zeros_in = length(intersect(F_index(P1+1:P),temp));
    s2 = s2 + zeros_in;
    
    temp1 = setdiff(F_index,temp);
    nonzeros_ex = length(intersect(F_index(1:P1),temp1));
    t2 = t2 + nonzeros_ex;
end
Mean2Zeros_In = s2/fre2;
Mean2NonZeros_Ex = t2/fre2;

s3 = 0;
t3 = 0;
for i = 1:fre3
    temp = IND3(i,:);
    zeros_in = length(intersect(F_index(P1+1:P),temp));
    s3 = s3 + zeros_in;
    
    temp1 = setdiff(F_index,temp);
    nonzeros_ex = length(intersect(F_index(1:P1),temp1));
    t3 = t3 + nonzeros_ex;
end
Mean3Zeros_In = s3/fre3;
Mean3NonZeros_Ex = t3/fre3;

MeanNonzeros_EX = [Mean1NonZeros_Ex;Mean2NonZeros_Ex;Mean3NonZeros_Ex];
MeanZeros_In = [Mean1Zeros_In;Mean2Zeros_In;Mean3Zeros_In];

%CER(classification error rate)
CLU1 = Clu(n1,:);
CLU2 = Clu(n2,:);
CLU3 = Clu(n3,:);

TrueCluResult = [10*ones(1,N1),20*ones(1,N2)];

s4 = 0;
for i = 1:fre1
    y = CER(CLU1(i,:),TrueCluResult);
    s4 = s4 + y;
end
CER1 = s4/fre1;

s5 = 0;
for i = 1:fre2
    y = CER(CLU2(i,:),TrueCluResult);
    s5 = s5 + y;
end
CER2 = s5/fre2;

s6 = 0;
for i = 1:fre3
    y = CER(CLU3(i,:),TrueCluResult);
    s6 = s6 + y;
end
CER3 = s4/fre3;

CERR = [CER1;CER2;CER3];

time = toc;
save l1.mat Fre MeanMeanLogLF StdMeanLogLF Meanlambda Stdlambda MeanNonzeros_EX MeanZeros_In CERR time;
clear;
load l1.mat;