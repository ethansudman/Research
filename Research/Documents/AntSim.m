% Amount that alpha decays when a forager leaves
q = 0.05;

% Baseline rate of foragers, in seconds
alpha = 0.01;

% "Natural" decay rat over time. Set to 0 in Gordon et. al., but other
% studies indicate that this may be significant.
d = 0;

% Set initial rates to 0

% an is lambda for Dn, which is a Poisson random variable
an = 0;

% Dn is the number of ants that depart at time n
%Dn = makedist('Poisson', 'lambda', 0);
Dn = poissrnd(an);


csv = [];

% An is the number of ants that arrive at time n
for An = 0.1:0.1:4
    % Amount that alpha increases when a forager returns
    c = 0.1;
    
    % Sweep between c = 0.1 and c = 0.25
    while (c <= 0.25)
        % Run the simulation for 200 iterations
        for n = 1:200
            %x = pdf(Dn, [an])
            an = max(an - (q * Dn) + (c * An) - d, alpha);

            Dn = poissrnd(an);
            %Dn = makedist('Poisson', 'lambda', an);
        end

        if 0.15 <= an && an <= 1.3
            csv = [csv; c An an Dn];
        end

        an = 0;
        Dn = poissrnd(an);

        c = c + 0.01;
    end
end

writematrix(csv, 'Corrected Simulation results.csv')