function [Mu,Pi,Sigma,CluResult,INDEX,LogLF] = EMforMPLE(X,X_Validate,K,lambda,penatype,tol,maxIter)

% August 2013
% This matlab code implements the EM Algorithm for evaluating the
% parameters of penalized log likelihood.
%
% Input:
% X - n1 x p matrix of observations,Training data
%   - X = [x1',x2',...,xn1']',  xi - p x 1 vector (i=1,2,...,n1)
% X_Validate - n2 x p matrix of observation, Validate data
% K - the number of compoents of mixture gaussian
% lambda - penalized parameter
% penatype - the type of penalized function
%          - here the choices of penatype are 'l0','l1','l0.5','SCAD','MCP'
% tol - tolerance for stopping criterion
%     - DEFAULT 1e-6
% maxIter - maximum number of iterations
%         - DEFAULT 300
%
% Output:
% Parameter needs to be estimate
% Mu - K x p matrix
%    - Mu = [mu1',mu2',...,muK']'
% Pi - K x 1 vector
%    - Pi= [pi1,pi2,...,piK]'
% Sigma - p x p matrix
%       - Sigma = diag(sigma1,sigma2,...,sigmaP)
% INDEX - the index of features which don't contribute to cluster
% 
% CluResult - the result of cluster
%
% LogLF - criterion for selecting models - log likelihood function
%
% Xiangyong Cao - caoxiangyong45@gmail.com
% Xiangyu Chang - xiangyuchang@gmail.com
% Copyright: Department of Statistics School of Mathematics and Statistics
% Xi'an Jiaotong University

if nargin < 6
    tol = 1e-6;
end

if nargin < 7
    maxIter = 300;
end


%Initialize(kmeans initialization)
[n,p] = size(X);
[Ind, C] = kmeans(X,K);

Pi = [];
for i = 1:K
    num =  length(find(Ind == i));
    Pi = [Pi;num];
end
Pi = Pi/n;

Mu = C;

% Sigma = diag(ones(p,1));
Sigma = diag(diag(cov(X)));

Tau = zeros(n,K);

INDEX = [];
Ori_p = p;
t = 0;
L = Comlikelihood(X,Pi,Mu,Sigma,Tau);
L1 = inf;

while t <= maxIter && abs(L-L1)>=tol
    
    L1 = L;
    
    [n,Newp] = size(X);
    p = Newp;
    
    %% E Step
    % Update Tau
    for i = 1:n
        Pr = Gaussian(X(i,:),Mu,Sigma);
        T = Pi.* Pr;
        Tau(i,:) = (T/sum(T))';
    end
    
    %% M Step
    % Update Pi
    nn = (sum(Tau))';
    Pi = nn./n;
    
    % Update Sigma
    S = zeros(p,p);
    for i = 1:n
        for k = 1:K
            S = S + Tau(i,k)*diag(diag((X(i,:)-Mu(k,:))'*(X(i,:)-Mu(k,:))));
        end
    end
    Sigma = S./n;
    
    % Update Mu
    for k = 1:K
        S1 = zeros(1,p);
        for i = 1:n
            S1 = S1 + Tau(i,k)*X(i,:);
        end
        mu_hat = (1/nn(k)) * S1;
        lambda_hat =  (lambda/nn(k)*Sigma*ones(p,1))';
        
        switch penatype
            case 'l1'
                Mu(k,:) = wthresh(mu_hat,'s',lambda_hat);      % L1 Penalty----Soft Threshoding
                
            case 'l0'
                Mu(k,:) = wthresh(mu_hat,'h',lambda_hat);      % L0 Penalty----Hard Threshoding
                
            case 'l0.5'
                Mu(k,:) = Half_Thresh(mu_hat,lambda_hat);       % L0.5 Penalty----Half Threshoding
                
            case 'SCAD'
                Mu(k,:) = SCAD_Thresh(mu_hat,lambda_hat);       % SCAD Penalty----SCAD Threshoding
                
            case 'MCP'
                Mu(k,:) = MCP_Thresh(mu_hat,lambda_hat);        % MCP Penalty----MCP Threshoding
            otherwise
                error('Invalid penality type or the penality type is not included.');
        end
        
    end
    
    %% Find out features which don't contribute to cluster
    index = [];
    
    q = length(INDEX);
    p1 = Ori_p - q;
    DiaSig = diag(Sigma)';
    for i = 1:q
        Mu = [Mu(:,1:INDEX(i)-1),zeros(K,1),Mu(:,INDEX(i):p1)];
        
        X = [X(:,1:INDEX(i)-1),zeros(n,1),X(:,INDEX(i):p1)];
        
        DiaSig = [DiaSig(1:INDEX(i)-1),zeros(1),DiaSig(INDEX(i):p1)];
        p1 = p1 + 1;
    end
    Sigma = diag(DiaSig);
    
    for j = 1:Ori_p
        [temp1,temp2] = find(Mu(:,j)==0);
        if sum(temp2)==K
            index = [index,j];
        end
    end
    
    INDEX = [];
    INDEX = [INDEX,index];  % record the subscript of  features which don't contribute to cluster
    
    X(:,INDEX) = [];    % remove the features which don't contribute to cluster
    
    Mu(:,INDEX) = [];
    
    diaS = diag(Sigma);
    diaS(INDEX) = [];
    Sigma = diag(diaS);
    t = t + 1;
    
    L = Comlikelihood(X,Pi,Mu,Sigma,Tau);
end

[maxValue,CluResult1] = max(Tau');
if K == 1
    CluResult = ones(1,n);
else
    CluResult = CluResult1;
end

% q1 = length(INDEX);
% de = K + p + K*p - 1 - q1;
% BIC = -2*L + log(n)*de;
X_Validate(:,INDEX) = [];
L_Validate = Comlikelihood(X_Validate,Pi,Mu,Sigma,Tau);
LogLF = -L_Validate;     


