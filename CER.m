% the function to calculate classification error rate(CER)
function y = CER(P,Q)
% P - the ture partition
% Q - the approximated partition
n = length(P);

s = 0;
for i = 1:n
    for j = i+1:n
      s = s + abs(Indicator(P(i),P(j))-Indicator(Q(i),Q(j)));    
    end
end
y =2*s/(n*(n-1));
