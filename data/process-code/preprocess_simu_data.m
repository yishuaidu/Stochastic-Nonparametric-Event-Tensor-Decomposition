clear all;
close all;

%pre-process simluation data
%find parents/children candidates, separate training/testing

load('./Simu.mat');
ind = Seq.ind;
ind_events = ind(Seq.Mark,:);
train_events = Seq.Time(1:1500);
test_events = Seq.Time(1501:end);
Dmax = 0.01;

tr = [];
tr.e = train_events;
tr.ind = ind_events(1:1500,:);
[par, child] = locate_driving_candidate(tr.e, Dmax);
tr.par = par;
tr.child = child;
data = tr;
data.T = tr.e(end);
save('Simu-train.mat', 'data');

te = [];
te.e = test_events;
te.ind = ind_events(1501:end,:);
[par, child] = locate_driving_candidate(te.e, Dmax);
te.par = par;
te.child = child;
data = te;
data.T1 = te.e(1);
data.T2 = te.e(end);
save('Simu-test.mat', 'data');

