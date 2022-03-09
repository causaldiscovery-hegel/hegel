clc; clear;
D1 = csvread('D.csv');
FDR = 0.01;
% Step 2
Y = D1(:,end); D1 = D1(:,1:end-1); [m,n1] = size(D1); % C=unique(cat(1,B{:}));
I2 = nchoosek(1:n1,2); n2 = size(I2,1); D2 = zeros(m,n2);
for i = 1:n2; D2(:,i) = D1(:,I2(i,1)).*D1(:,I2(i,2)); end
n0 = n1+n2+1; D = [D1 D2 Y]; % I = de2bi([0:(2^n1-1)],'left-msb');
I = Causal_Explorer('GS',D,n0,2*ones(1,n0),'g2',0.1);
n = length(I); X = [D(:,I) D(:,end)]; ID = zeros(n,2);
for i = 1:n; if I(i)<=n1; ID(i,1) = I(i); else; ID(i,:) = I2(I(i)-n1,:); end; end
%[I,~,~,~,~]=Causal_Explorer('MMHC',D,ones(1,n0),'GreedySearch',[],10,'BDeu');

% Step 3 (ID, X, n)
nHyp = n0; P = ones(n,1);
L = pdist2(X,X,'cityblock');
for i = 1:n
    x = X(:,i); U = find(x); V = find(1-x); Li = L(U,V);
    [P1,P2] = ind2sub(size(Li),find(munkres(Li)));
    if isempty(P1)==0
    x2 = P1; x1 = U'; z = 1:length(U);
    for j = 1:numel(x1); x2(P1 == z(j)) = x1(j); end; P1 = x2;
    x2 = P2; x1 = V'; z = 1:length(V);
    for j = 1:numel(x1); x2(P2 == z(j)) = x1(j); end; P2 = x2;
    T = zeros(2,2);
    for j = 1:length(P1); T = T+[Y(P1(j));1-Y(P1(j))]*[Y(P2(j)) 1-Y(P2(j))]; end
    T1 = T(1,2); T2 = T(2,1); p = 1 - chi2cdf((T1-T2)^2/(T1+T2),1); p(p==0) = 1;
    P(i) = p;
    end
end
loglog(1:n, (FDR/nHyp)*[1:n], 1:n, sort(P), '.-');
xlim([1 n]); grid minor; xlabel('Potential Cause'); ylabel('P-value');
legend('BH Baseline','Hypotheses','Location','northwest')
Results = table(ID,P)