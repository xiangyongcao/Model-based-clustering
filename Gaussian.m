%%Gaussian distribution
function Pr = Gaussian(x,Mu,Sigma)

% x - p x 1 vector
% Mu - K*p matrix
% Sigma - p x p  matrix of diag

[K p] = size(Mu);
Pr = zeros(K,1);
for i = 1:K
    Pr(i) =1/det(Sigma)^(1/2)*exp(-0.5*(x-Mu(i,:))*inv(Sigma)*(x-Mu(i,:))');
end