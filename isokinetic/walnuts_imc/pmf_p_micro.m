function prob = pmf_p_micro(j, i)
    weights = [2/3, 1/3, 0];
    if ismember(j, [i, i+1, i+2])
        prob = weights(j - i + 1);
    else
        prob = 0;
    end
end