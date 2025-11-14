function result = sub_u_turn(O)
    % Recursive U-turn check on orbit segments.
    % O is an N-by-m cell array of states (theta, velocity, [W, ...]).

    n = size(O, 1);   % # of states = # of rows
    if n < 2
        result = false;
        return;
    end

    mid = floor(n / 2);

    result = u_turn(O) ...
          || sub_u_turn(O(1:mid,   :)) ...
          || sub_u_turn(O(mid+1:end, :));
end