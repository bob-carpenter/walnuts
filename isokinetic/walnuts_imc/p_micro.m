function j = p_micro(i)
    weights = [2/3, 1/3, 0];
    choices = [i, i+1, i+2];
    cdf = cumsum(weights);
    r = rand() * cdf(end);
    j = choices(find(cdf >= r, 1, 'first'));
end