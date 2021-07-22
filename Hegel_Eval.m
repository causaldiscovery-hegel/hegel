%% Parameterization
clc; format short g; clear variables; close all; warning off;
Data_Directory = 'Hegel (Subgroup Discovery)/Data/';
n0 = 10; FDR = 0.01;
Index_Shift = 0;

%% Importing
[PZ,~] = xlsread('Data.xlsx');
NGC = readtable([Data_Directory 'Subgroups/Depth 1.txt']);
[~,NGA] = xlsread([Data_Directory 'NG1.xlsx']);
PGA = csvread([Data_Directory 'PG1.csv']);
disp('Imported ...')

%% Preprocessing
nP = size(PZ,1); nGC = size(NGC,1); nGA = length(NGA);
PGA = reshape(PGA,[nGA,nP])';
x = PZ(:,5); PZ(:,5) = (x-min(x))/(max(x)-min(x));
x = PZ(:,6); PZ(:,6) = (x-min(x))/(max(x)-min(x));
PR = PZ(:,3); PZ = PZ(:,4:10); DPP = pdist2(PZ,PZ,'cityblock');
GCAI = zeros(1,nGC); UGC = zeros(1,nGC);
for i = 1:nGC
    NGCi = NGC{i,2}; NGCi = NGCi{1};
    UGC(i) = str2num(NGCi(end));
    GCAI(i) = find(strcmp(NGCi(1:end-2),NGA)) + Index_Shift;
end
PGC = PGA(:,GCAI);
for i = 1:nGC; if UGC(i)==0; PGC(:,i) = 1-PGC(:,i); end; end
disp('Pre-processed ...')

%% P-Values
P_values = ones(nGC,1); Co_Prevalences = zeros(nGC,1);
for i = 1:nGC
    PGi = PGC(:,i); U = find(PGi); V = find(1-PGi); dUV = DPP(U,V);
    [P1,P2] = ind2sub(size(dUV),find(munkres(dUV)));
    if isempty(P1)==0
    x2 = P1; x1 = U'; x = 1:length(U);
    for j = 1:numel(x1); x2(P1 == x(j)) = x1(j); end; P1 = x2;
    x2 = P2; x1 = V'; x = 1:length(V);
    for j = 1:numel(x1); x2(P2 == x(j)) = x1(j); end; P2 = x2;
    T = zeros(2,2);
    for j = 1:length(P1); T = T+[PR(P1(j));1-PR(P1(j))]*[PR(P2(j)) 1-PR(P2(j))]; end
    T1 = T(1,2); T2 = T(2,1);
    P = 1 - chi2cdf((T1 - T2)^2 / (T1 + T2),1); P(P==0) = 1;
    Co_Prevalences(i) = PR'*PGi; P_values(i) = P;
    end
end
Scores = table2array(NGC(:,1));
Correlations = corr(PGC,PR);
SNPs = NGC(:,1);
disp('Analyzed')

%% Results
% Candidates = [table(SNPs,Scores,P_values,Co_Prevalences,Correlations)];
% Candidates = sortrows(Candidates,3);
nHypothesis = sum(Co_Prevalences>=n0);
figure(1);
loglog(1:nGC, (FDR/nHypothesis)*[1:nGC], 1:nGC, sort(P_values), '.-');
xlim([1 nGC]); grid minor; xlabel('SNP'); ylabel('P Value');
legend('BH Baseline','Hypotheses','Location','northwest')
P_values(find(isnan(P_values))) = 1;
Score_Trend = cumsum(P_values)./[1:nGC]';
figure(2); semilogx(Score_Trend, 'LineWidth',2)
xlim([1 nGC]); grid minor; xlabel('Ranking in SD'); ylabel('P Value');

