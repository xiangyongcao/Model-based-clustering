%%The lower bound of Penalized Complete Likelihood function
function L = Comlikelihood(X,Pi,Mu,Sigma,Tau)

[n,p] = size(X);
[K,p] = size(Mu);

y1 = sum(Tau)*log(Pi);

y2 = 0;
for k = 1:K
  for j =1:n
     Pr = Gaussian(X(j,:),Mu(k,:),Sigma);
     y2 = y2 + Tau(j,k)*Pr;
  end
end

L = y1 + y2;
