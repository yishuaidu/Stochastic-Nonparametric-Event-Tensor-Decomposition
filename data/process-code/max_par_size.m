function max_par = max_par_size(file_name)
    load(file_name);
    max_par = 0;
    for i=1:length(data.par)
        if max_par < length(data.par{i})
            max_par = length(data.par{i});
        end
    end
    
end