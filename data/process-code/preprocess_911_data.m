clear all;
close all;

%pre-process simluation data
%find parents/children candidates, separate training/testing

load('../911.mat');
ind_events = datas.ind;
T = datas.T;
events = datas.e;
events = events/3600;
T = T/3600;
delim = 40000;
test_sz = 10000;
Dmax = 2;
train_events = events(1:delim);
test_events = events(delim+1:delim+test_sz);

tr = [];
tr.e = train_events;
tr.ind = ind_events(1:delim,:);
[par, child] = locate_driving_candidate(tr.e, Dmax);
tr.par = par;
tr.child = child;
data = tr;
data.T = tr.e(end);
save('../911-train.mat', 'data');

te = [];
te.e = test_events;
te.ind = ind_events(delim+1:delim+test_sz,:);
[par, child] = locate_driving_candidate(te.e, Dmax);
te.par = par;
te.child = child;
data = te;
data.T1 = te.e(1);
data.T2 = te.e(end);
save('../911-test.mat', 'data');

