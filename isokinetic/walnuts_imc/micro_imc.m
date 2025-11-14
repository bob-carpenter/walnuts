function ell = micro_imc(logmu, gradlogmu, theta0, rho0, h0, delta, K0)
% Choose ell so that, with step h = h0 / 2^ell and R = 2^ell BAB-steps,
% the range of logmu(theta) - W along the path is <= delta.

    ell = 0;


    while true
        theta = theta0;
        rho   = rho0;
        W     = 0;

        R = 2^ell;
        h = h0 / R;

        % initial modified "energy"
        E = -logmu(theta) + W;
        E_max = E;
        E_min = E;

        % Flag to record whether this ell was numerically usable
        bad_ell = false;

        for j = 1:R
            % one BAB-IMC micro-step
            [theta, rho, dW] = bab_isokinetic(logmu, gradlogmu, theta, rho, h, 1, K0);
            W = W + dW;

            E = -logmu(theta) + W;

            if ~isfinite(E)
                % Numerical blow-up: abandon this ell and try a finer step
                bad_ell = true;
                break;
            end

            E_max = max(E_max, E);
            E_min = min(E_min, E);
        end

        % If this ell was numerically bad, refine h by increasing ell
        if bad_ell
            ell = ell + 1;
            continue;
        end

        if E_max - E_min <= delta
            return;
        end

        ell = ell + 1;
    end
end