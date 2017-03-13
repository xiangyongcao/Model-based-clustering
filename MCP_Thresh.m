%MCP Threshoding
function y = MCP_Thresh(x,lambda)

%The function solve the optimization:
% min 1/2*(x(i)-y)^2 + lambda*p_mcp(|y|)
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
gama = 2;   % ?? gama>1  gama to 1 : Hard    gama to inf: Soft

for i = 1:p
    if abs(x(i))<=gama*lambda(i)
        y(i) = wthresh(x(i),'s',lambda(i))/(1-1/gama);
    else
        y(i) = x(i);
    end
end