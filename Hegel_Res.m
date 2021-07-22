clc; clear variables; close all;
Parameter = {'m','n','q','z','l1','l2'}';
Vals = [500 1000 5000 10000 ; 125 250 500 1000
    .03 .05 .1 .2 ; .25 .5 .75 1
    0 1 2 3 ; 1 2 3 4];
Acc1 = zeros(size(Vals)); AccDen = 0;
for k = 1:length(Parameter)
Par = Parameter{k}; Val = Vals(k,:);
for j = 1:length(Val)
Fold = ['finalResult\' Par '\' num2str(Val(j)) '\'];
D=dir(Fold); D=D(~ismember({D.name},{'.','..'})); Tests = size(D,1);
if k<5
for i = 1:Tests
File = D(i).name;
v = fileread([Fold File '\true_causes.txt']);
U = fileread([Fold File '\beam.txt']);
Acc1(k,j) = Acc1(k,j) + contains(U,v)/Tests;
end
else
for i = 1:Tests
File = D(i).name;
v = fileread([Fold File '\true_causes.txt']);
U = fileread([Fold File '\beam.txt']);
U1 = split(U,'; '); v1 = split(v,'; '); v1 = v1(~cellfun('isempty',v1));
Acc1(k,j) = Acc1(k,j)+ numel(intersect(v1,U1)); AccDen = AccDen+numel(v1);
end
Acc1(k,j) = Acc1(k,j)/AccDen; AccDen = 0;
end
end
end

%% Present
Names = {'(a)','(b)','(c)','(d)'};
VarName = {'Number of Features (n)','Number of Samples (m)',...
    'Rate of Noise (q)','Confounders Satisfaction (z)'};
Acc1 = round(Acc1*100)/100; disp(table(Parameter,Acc1))
Vals(3,1) = 0.025;
%%
Acc2 = csvread('B.csv'); Acc3 = csvread('C.csv'); AccF = csvread('F.csv');
close all; figure;
for i = 1:4
    subplot(2,2,i); hold on;
    plot(Vals(i,:),Acc1(i,:),'s--','MarkerSize',5);
    plot(Vals(i,:),Acc2(i,:),'o-.','MarkerSize',4);
    plot(Vals(i,:),Acc3(i,:),'.-','MarkerSize',12);
    plot(Vals(i,:),AccF(i,:),'+:','MarkerSize',6);
    if i<4; set(gca, 'XScale', 'log'); end
    ylim([0 1]); xlim([min(Vals(i,:)) max(Vals(i,:))]); xticks(Vals(i,:));
    title(Names{i}); grid minor; xlabel(VarName(i));
    if i==1; legend('SD','SD+SL','SD+SL+QED', 'FCI','Location',...
            'southwest','NumColumns',2); end
    if i==1||i==3; ylabel('Accuracy'); yticks([0 .2 .4 .6 .8 1]);
    else; yticklabels([]); end
end
