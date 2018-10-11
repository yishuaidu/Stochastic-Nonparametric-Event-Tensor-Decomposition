clear all;
close all;

%pre-process simluation data
%find parents/children candidates, separate training/testing

load('./Simu-large.mat');
tr_num = 8000;
ind = Seq.ind;
ind_events = ind(Seq.Mark,:);
train_events = Seq.Time(1:tr_num);
test_events = Seq.Time(tr_num+1:end);
Dmax = 0.01;

tr = [];
tr.e = train_events;
tr.ind = ind_events(1:tr_num,:);
[par, child] = locate_driving_candidate(tr.e, Dmax);
tr.par = par;
tr.child = child;
data = tr;
data.T = tr.e(end);
save('Simu-train-large.mat', 'data');

te = [];
te.e = test_events;
te.ind = ind_events(tr_num+1:end,:);
[par, child] = locate_driving_candidate(te.e, Dmax);
te.par = par;
te.child = child;
data = te;
data.T1 = te.e(1);
data.T2 = te.e(end);
save('Simu-test-large.mat', 'data');

