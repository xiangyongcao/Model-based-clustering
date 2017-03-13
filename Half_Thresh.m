%Half Threshoding
function y = Half_Thresh(x,lambda)

% The function solve the optimization:  
% min 1/2*(x(i)-y)^2 + lambda*|y|^0.5
% x - p x 1 vector  
% lambda - scalar or vector

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
Phix = acos(lambda./8.*(abs(x)/3).^(-1.5));

for i = 1:p
    if abs(x(i))>54^(1/3)/4*lambda(i)^(2/3)
        y(i) = 2/3*x(i)*(1+cos(2*pi/3-2*Phix(i)/3));
    else
        y(i) = 0;
    end
end
