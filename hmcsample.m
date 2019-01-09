function [samples, accept] = hmcsample(varargin)
%HMCSAMPLE Hamiltonian Monte-Carlo sampler. Two distinct modes are available: first is standard HMC
%in which the particle is given a constant mass. The second mode is Riemannian HMC in which the
%"mass" is reinterpreted as a local metric in the differential-geometric sense, and is allowed to
%change across the space in order to exploit local geometry and make the result invariatnt to
%reparameterizations. No checks are done on particle bounds, so bounded parameters should be
%transformed to continuous ones first (and the metric should be adjusted accordingly).
%
%My reference for Riemannian HMC is Girolami et al (2004). Comments citing [GCC04] are referring to
%equations in this paper. It can be found here:
%https://pdfs.semanticscholar.org/16c5/06c5bb253f7528ddcc80c72673fabf584f32.pdf
%
%Standard HMC is enabled by default. Supplying a 'Metric' argument enables the Riemann version. All
%function handles should accept and return column vectors where appropriate.
%
% [samples, accept] = HMCSAMPLE(startPoint, logPDF, gradLogPDF) takes as input the starting point (a
% [column vector) and two function handles that compute the log density and its gradient. Returns an
% [s x d] matrix of samples (each is dimension d) and the acceptance probability.
%
% HMCSAMPLE(..., 'PropSteps', numSteps) sets the number of discrete steps to run the differential
% equations forward for each proposal (default 25)
%
% HMCSAMPLE(..., 'StepSize', stepSize) sets how to scale the gradient for each step (default 0.01)
%
% HMCSAMPLE(..., 'NSamples', nSamples) sets the total number of samples to draw (default 1000)
%
% HMCSAMPLE(..., 'BurnIn', burnIn) sets the number of initial samples to throw away (default 100)
%
% HMCSAMPLE(..., 'Mass', M) sets the mass matrix (if not Riemannian mode) (default identity matrix)
%
% HMCSAMPLE(..., 'CheckGradients', flag) whether or not to numerically verify gradLogPDF during
% burn-in (default false)
%
% HMCSAMPLE(..., 'Metric', metricFn) a function handle taking a sample x as input and returning the
% local metric at that point. If set, this automatically enables Riemannian mode.
%
% HMCSAMPLE(..., 'GradMetric', gradMetricFn) (optional) a function handle giving the gradient of the
% metric itself. The function should take two arguments (x, i) and return a [d x d] matrix of how
% metricFn(x) changes with respect to x(i). If not specified, a central-difference numeric
% approximation is used.

%% Parse inputs
p = inputParser;
p.FunctionName = 'hmcsample';
p.addRequired('startpoint');
p.addRequired('logpdf');
p.addRequired('gradlogpdf');
p.addParameter('propsteps', 25, @(x) x >= 1);
p.addParameter('stepsize', 0.01, @(x) isscalar(x) && x > 0);
p.addParameter('nsamples', 1000, @(x) x >= 1);
p.addParameter('burnin', 100, @(x) x >= 0);
p.addParameter('mass', []);
p.addParameter('checkgradients', false, @islogical);
p.addParameter('metric', []);
p.addParameter('gradmetric', []); % @(x, i) --> [d x d] matrix dG/dx_i
p.parse(varargin{:});
args = p.Results;

% Dimensionality of space is 'd'
d = length(args.startpoint);

% The riemann integrator is automatically turned on if a metric is given.
riemann = ~isempty(args.metric);

if isempty(args.mass)
    % Default mass is identity matrix
    args.mass = eye(d);
elseif riemann
    warning('Using riemann metric - mass matrix will be ignored! (but I''ll adjust the step size accordingly)');
    avg_mass = det(args.mass)^(1/d);
    args.stepsize = args.stepsize / avg_mass;
end

%% Define function to get and update local metric information relatively efficiently

    function [G, Gi, cholG, dGdx] = metric(x)
        if riemann
            G = args.metric(x);
        else
            G = args.mass;
        end
        if nargout >= 2, Gi = inv(G); end
        if nargout >= 3, cholG = chol(G); end
        if nargout >= 4
            if ~isempty(args.gradmetric)
                for i=d:-1:1
                dGdx(:,:,i) =  args.gradmetric(x, i);
                end
            else
                % Numerically approximate dGdx, which is a [d x d x d] tensor where dGdx(:,:,i)
                % contains dG_dxi
                dGdx = numericgrad(@metric, x);
            end
        end
    end

[~, Gi, cholG, dGdx] = metric(args.startpoint);

%% Define Hamiltonian

    function H = Hamiltonian(x, p)
        % The full hamiltonian, up to a constant (log 2pi). Note that values will change as Gi and
        % rootG are changed outside the scope of this function.
        G = metric(x);
        logdet_G = sum(log(diag(chol(G))));
        H = -args.logpdf(x) + 1/2*(logdet_G + p'*(G\p)); % [GCC04] equations (3) and (19)
    end

    function [x, p, phalf] = LeapfrogUpdate(x, p)
        % [GCC04] equations (5)-(7)
        phalf = p + args.stepsize * args.gradlogpdf(x) / 2;
        x = x + args.stepsize * Gi * phalf;
        p = phalf + args.stepsize * args.gradlogpdf(x) / 2;
    end

    function pnhalf = GeneralizedLeapfrogMomentum(xn, pn, pnhalf)
        % [GCC04] equations (21) and (22)
        dLdx = args.gradlogpdf(xn);
        dHdx = zeros(size(xn));
        Gip = Gi*pnhalf;
        for i=1:d
            dHdx(i) = -dLdx(i) + 1/2*tracemul(Gi, dGdx(:,:,i)) - 1/2*Gip'*dGdx(:,:,i)*Gip;
        end
        pnhalf = pn - args.stepsize/2 * dHdx;
    end

    function x_next = GeneralizedLeapfrogX(xn, phalf, x_next)
        % [GCC04] equations (20) and (23)
        G_next = metric(x_next);
        dHdp = Gi * phalf;
        dHdp_next = G_next \ phalf;
        x_next = xn + args.stepsize * (dHdp + dHdp_next) / 2;
    end

%% Initialize results
nTotalSamples = args.nsamples + args.burnin + 1;
samples = zeros(nTotalSamples, d);
samples(1, :) = args.startpoint;
didAccept = zeros(nTotalSamples, 1);

%% Loop to generate samples
for iSample=2:nTotalSamples
    % Check user-supplied gradients during burn-in if requested
    if args.checkgradients && iSample <= 1+args.burnin
        [adiff, rdiff] = checkgradient(args.logpdf, args.gradlogpdf, samples(iSample-1, :)');
        fprintf('Sample %d: x = [%s]\tabs grad diff = %.2e\trel grad diff = %.2e\n', iSample-1, num2str(samples(iSample-1, :), 2), adiff, rdiff);
    end
    
    % Start sample point x where previous sample left off
    x = samples(iSample-1, :)';
    % Draw new momentum value taking local geometry into account (move quickly in directions where
    % the metric defines changes as happening slowly and vice versa)
    p = cholG * randn(d, 1);
    % For use in accept/reject, we need to know the log probability of this starting point
    prev_logpdf = -Hamiltonian(x, p);
    
    % Simulate forward
    for iStep=1:args.propsteps
        if ~riemann
            % In the non-riemannian case, we use the standard leapfrog integrator
            [x, p] = LeapfrogUpdate(x, p);
        else
            % In the case where a riemannian metric is supplied, use the Generalize Leapfrog
            % algorithm to ensure it is volume-preserving
            phalf = p;
            for iFixedPoint=1:10
                phalf = GeneralizedLeapfrogMomentum(x, p, phalf);
            end
            xnext = x;
            for iFixedPoint=1:10
                xnext = GeneralizedLeapfrogX(x, phalf, xnext);
            end
            pnext = GeneralizedLeapfrogMomentum(x, phalf, phalf);
            
            % Update
            x = xnext;
            p = pnext;
        end
    end
    
    % Compute joint (x,p) probability at endpoint
    new_logpdf = -Hamiltonian(x, p);
    
    % Compute the Metropolis-Hastings acceptance probability
    mhratio = min(1, exp(new_logpdf - prev_logpdf));
    
    if rand < mhratio
        % Accept case
        didAccept(iSample) = true;
        samples(iSample, :) = x;
        % Need to update the shared variables Gi, cholG, and dGdx now that x has changed
        [~, Gi, cholG, dGdx] = metric(x);
    else
        % Reject case
        didAccept(iSample) = false;
        samples(iSample, :) = samples(iSample-1, :);
    end
end

%% Post-process

% Throw away burn-in samples
samples = samples(args.burnin+2:end, :);
% Compute final acceptance probability (excluding burn-in)
accept = mean(didAccept(args.burnin+2:end));

end

%% Helper functions

function v = tracemul(A, B)
% Faster computation of trace(A*B)
AB = A' .* B;
v = sum(AB(:));
end

function [adiff, rdiff] = checkgradient(f, gradf, x)
num = numericgrad(f, x);
ana = gradf(x);

adiff = max(abs(num(:) - ana(:)));
rdiff = max(abs(num(:) - ana(:)) ./ abs(num(:)));
end

function grad = numericgrad(f, x)
d = length(x);
out = f(x);
if isscalar(out)
    grad = zeros(d, 1);
    S = struct('type', '()', 'subs', {{1}});
else
    grad = zeros([size(out) d]);
    S = struct('type', '()', 'subs', {repmat({':'}, 1, ndims(grad))});
end
eps = 1e-7;
for i=1:d
    delta = zeros(size(x));
    delta(i) = eps;
    S.subs{end} = i;
    grad = subsasgn(grad, S, (f(x + delta) - f(x - delta)) / (2 * eps));
end
end