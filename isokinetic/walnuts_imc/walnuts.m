function [theta_tilde, T] = walnuts(logmu, gradlogmu, theta, h, i_max, delta)
    d = length(theta);
    rho = randn(d, 1);
    theta_tilde = theta;
    rho_tilde = rho;
    logw_0 = logmu(theta) - 0.5 * norm(rho)^2;
    
    O = {theta, rho};
    logW = logw_0;
    B = randi([0, 1], i_max, 1);
    
    for i = 1:i_max
        O_old = O;
        logW_old = logW;
        
        if B(i) == 1
            [O_ext, logW_ext] = extend_orbit_forward(logmu, gradlogmu, O{end,1}, O{end,2}, logW(end), h, delta, 2^(i-1));
            O = [O; O_ext];
            logW = [logW; logW_ext];
        else
            [O_ext, logW_ext] = extend_orbit_backward(logmu, gradlogmu, O{1,1}, O{1,2}, logW(1), h, delta, 2^(i-1));
            O = [O_ext; O];
            logW = [logW_ext; logW];
        end
        
        if sub_u_turn(O_ext)
            break;
        end
        
        logu = log(rand);
        
        if logu <= logsumexp(logW_ext) - logsumexp(logW_old)
            weights = exp(log_softmax(logW_ext));
            cdf = cumsum(weights);
            r = rand() * cdf(end);
            idx = find(cdf >= r, 1, 'first');
            theta_tilde = O_ext{idx, 1};
            rho_tilde = O_ext{idx, 2};
        end
        
        if u_turn(O)
            break;
        end
    end
    
    T = length(logW) * h;
end