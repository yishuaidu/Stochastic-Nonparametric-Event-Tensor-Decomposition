%identity paraents/children for an event sequence
%those at most Kmax events within Dmax ahead are treated as parents
function [par, child] = locate_driving_candidate_hybrid(events, Dmax, Kmax)
    n = length(events);
    par = cell(n,1);
    child = cell(n,1);
    for i=1:n
        par{i} = [];
        child{i} = [];
    end
    for i=1:n
        for j=i+1:min(i+Kmax,n)
            if abs(events(j) - events(i))<=Dmax
                if events(j)-events(i)<0
                    fprintf('Wrong data!\n');
                end
                child{i} = [child{i};j];
                par{j} = [par{j};i];
            else
                break;
            end
        end
    end
end