%identity paraents/children for an event sequence
%those at most Kmax events ahead are treated as parents
function [par, child] = locate_driving_candidate_nearest(events, Kmax)
    n = length(events);
    par = cell(n,1);
    child = cell(n,1);
    for i=1:n
        par{i} = [];
        child{i} = [];
    end
    for i=1:n
        for j=i+1:min(i+Kmax,n)
            par{j} = [par{j};i];
            child{i} = [child{i};j];
        end
    end
end