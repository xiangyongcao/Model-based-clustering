%SCAD Threshoding
function y = SCAD_Thresh(x,lambda)

% The function solve the optimization: 
% min 1/2*(x(i)-y)^2 + lambda*p_scad(|y|)
% x - p x 1 vector  
% lambda - vector or scalar

p = length(x);
n = length(lambda);
if n == 1
    lambda = repmat(lambda,1,p);
elseif n == p
    lambda = lambda;
else
    error('x and lambda should have the same dimension!')
end

y = rand(1,p);
a = 3.7;      % based on Bayesian argument

for i = 1:p
    if abs(x(i))<=2*lambda(i)
        y(i) = wthresh(x(i),'s',lambda(i));
    elseif abs(x(i))>2*lambda(i) && abs(x(i))<=a*lambda(i)
        y(i) = ((a-1)*x(i)-sign(x(i))*a*lambda(i))/(a-2);
    else
        y(i) = x(i);
    end
end