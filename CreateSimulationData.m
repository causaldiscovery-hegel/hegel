clc; clear variables;
m = 1000;% number of features
n = 500; % number of samples
l1 = 3;	 % number of singular causes
l2 = 2;	 % number of pair causes
p = 1/4; % non-sparsity of signal
z = 3/4; % non-sparsity of necessary confounders
q = 0.05;% rate of noise
Sp = 3/4;% distribution mean of Prior Score
%% Data Generation
X = rand(n,m); X = (X>1-p); y = zeros(n,1); % Features
v = datasample(1:m,1*l1+2*l2,'Replace',false); V = -ones(n,l1+l2); % Cause
for i = 1:l1 % Outcome
    x = X(:,v(i)); V(:,i) = x; y = y|x;
end
for i = 1:l2
    ii = l1+2*i; x = X(:,v(ii-1)).*X(:,v(ii)); V(:,l1+i) = x; y = y|x;
end
V1 = [v(1:l1)' v(1:l1)']; V2 = sort(reshape(v(l1+1:l1+2*l2),2,[])',2);
x = rand(n,1); x = (x>1-z); y = y&x; % Confounders
x = datasample(1:n,round(n*q),'Replace',false); y(x)=1-y(x); % Noise
S1 = 0.1*randn(1,m)+0.5; S1(S1>1)=1; S1(S1<0)=0; S1(v) = Sp; % 1D Prior
S2 = abs(corr(X)); % 2D Similarity
disp([num2str(l1) ' single & ' num2str(l2) ' pair causes: '])
vv  = [V1;V2]; disp(vv)
x = input('Export to CSV? [1/0]'); % Export
if x
    csvwrite('W.csv',S2); csvwrite('V.csv',S1);
    csvwrite('y.csv',y); csvwrite('X.csv',X); csvwrite('vv.csv',vv);
end