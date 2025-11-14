function [O, logW] = extend_orbit_backward(logmu, gradlogmu, ...
                                            theta_a, rho_a, W_a, logw_a, ...
                                            h, delta, L, K0)
% Extend orbit backward from boundary state (theta_a, rho_a, W_a, logw_a)
% using IMC-BAB micro-steps and WALNUTS random micro-times.

    O    = {};
    logW = [];

    theta = theta_a;
    rho   = rho_a;
    W     = W_a;
    logw  = logw_a;

    for iter = 1:L
        % backward micro-time selection (start with reversed rho)
        ell_b = micro_imc(logmu, gradlogmu, theta, -rho, h, delta, K0);
        ell   = p_micro(ell_b);

        % integrate backward in velocity (theta, -rho)
        h_eff = h * 2^(-ell);
        [theta_1, rho_tmp, dW] = bab_isokinetic(logmu, gradlogmu, theta, -rho, h_eff, 2^ell, K0);

        % flip rho back
        rho_1 = -rho_tmp;
        W_1   = W + dW;

        % forward micro-time from the new state
        
        ell_f = micro_imc(logmu, gradlogmu, theta_1, rho_1, h, delta, K0);
        % weight update in terms of logmu - W
        p_num = pmf_p_micro(ell, ell_f);
        p_den = pmf_p_micro(ell, ell_b);

        if p_num == 0
            logw = -Inf;
        else
            logpi_new = logmu(theta_1) - W_1;
            logpi_old = logmu(theta)   - W;

            logw = logw ...
                 + (logpi_new - logpi_old) ...
                 + log(p_num) - log(p_den);
        end

        % advance boundary state (prepended to orbit since it's backward)
        theta = theta_1;
        rho   = rho_1;
        W     = W_1;

        O    = [{theta, rho, W}; O];
        logW = [logw; logW];
    end
end