function [theta_tilde, T] = iwalnuts(logmu, gradlogmu, theta, h, i_max, delta, K0)
% WALNUTS with isokinetic BAB integrator (IMC version)
%
%   logmu       : handle, log target density logmu(theta)
%   gradlogmu   : handle, gradient of logmu
%   theta       : initial position
%   h           : base time step for BAB
%   i_max       : maximum tree depth (number of doublings)
%   delta       : micro-time energy window
%   K0          : kinetic energy parameter (||rho||^2 = 2 K0)
%
% Output:
%   theta_tilde : new position after one WALNUTS update
%   T           : total trajectory time (approx. length(logW)*h)

    d = length(theta);

    if nargin < 7
        % Default "isokinetic" choice (special incompressible case)
        K0 = (d - 1) / 2;
    end

    % Sample initial direction on the sphere: ||rho||^2 = 2 K0
    u = randn(d, 1);
    u = u / norm(u);
    rho = sqrt(2 * K0) * u;

    % Initial compressibility term
    W0 = 0;

    theta_tilde = theta;
    rho_tilde   = rho;

    % Initial log weight: logpi(theta, rho, W) = logmu(theta) - W
    logw_0 = logmu(theta) - W0;

    % Orbit and weights: store {theta, rho, W}
    O    = {theta, rho, W0};
    logW = logw_0;

    % Random doubling directions
    B = randi([0, 1], i_max, 1);

    for i = 1:i_max
        O_old   = O;
        logW_old = logW;

        if B(i) == 1
            % Extend forward from the right boundary
            [O_ext, logW_ext] = extend_orbit_forward( ...
                logmu, gradlogmu, ...
                O{end, 1}, O{end, 2}, O{end, 3}, logW(end), ...
                h, delta, 2^(i - 1), K0);

            O    = [O;    O_ext];
            logW = [logW; logW_ext];
        else
            % Extend backward from the left boundary
            [O_ext, logW_ext] = extend_orbit_backward( ...
                logmu, gradlogmu, ...
                O{1, 1}, O{1, 2}, O{1, 3}, logW(1), ...
                h, delta, 2^(i - 1), K0);

            O    = [O_ext;    O];
            logW = [logW_ext; logW];
        end

        % If the new subtree itself violates the local U-turn condition, stop
        if sub_u_turn(O_ext)
            break;
        end

        % Global orbit weighting: decide whether to select from O_ext
        logu = log(rand);

        if logu <= logsumexp(logW_ext) - logsumexp(logW_old)
            weights = exp(log_softmax(logW_ext));
            cdf = cumsum(weights);
            r = rand() * cdf(end);
            idx = find(cdf >= r, 1, 'first');

            theta_tilde = O_ext{idx, 1};
            rho_tilde   = O_ext{idx, 2};
            % W at that point is O_ext{idx, 3}, but we do not need it
            % because we re-draw a fresh direction at the next WALNUTS step.
        end

        % Stop if the full orbit satisfies the (global) U-turn condition
        if u_turn(O)
            break;
        end
    end

    % Approximate total trajectory time
    T = length(logW) * h;
end