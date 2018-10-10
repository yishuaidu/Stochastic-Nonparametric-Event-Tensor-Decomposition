clear all;
close all;

%pre-process simluation data
%find parents/children candidates, separate training/testing

load('../article_user.mat');
ind_events = datas.ind;
%indices starting from 1
ind_events = ind_events + 1; 


T = datas.T;
events = datas.e;
%day = 3600*24;
hour = 3600;
%events = events/day;
%T = T/day;
events = events/hour;
T = T/hour;
delim = 50000;
test_sz = 10000;
Dmax = 3; %maximum 1,2,3
Kmax = 300;

train_events = events(1:delim);
test_events = events(delim+1:delim+test_sz);


tr = [];
tr.e = train_events;
tr.ind = ind_events(1:delim,:);
[par, child] = locate_driving_candidate_hybrid(tr.e, Dmax, Kmax);
tr.par = par;
tr.child = child;
data = tr;
data.T = tr.e(end);
data.Dmax = Dmax;
data.Kmax = Kmax;
file_name_tr = strcat('../article-train-hybrid-', num2str(Dmax),'-',num2str(Kmax),'.mat');
save(file_name_tr, 'data');


te = [];
te.e = test_events;
te.ind = ind_events(delim+1:delim+test_sz,:);

[par, child] = locate_driving_candidate_hybrid(te.e, Dmax, Kmax);
te.par = par;
te.child = child;
data = te;
data.T1 = te.e(1);
data.T2 = te.e(end);
file_name_te = strcat('../article-test-hybrid-', num2str(Dmax),'-',num2str(Kmax),'.mat');
save(file_name_te, 'data');

fprintf('train: max # parents = %d\n', max_par_size(file_name_tr));
fprintf('test: max # parents = %d\n', max_par_size(file_name_te));