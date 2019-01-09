% Draw samples from a 2D correlated Gaussian. Assert that, using the same seed, the samples exactly
% match between Riemann-metric-based and mass-based samplers when the metric and mass are the same
% (when the number of steps and step size are reasonably small)
cov = [1 .8; .8 1];
cholCov = chol(cov);

startpoint = [0 0]';
logPDF = @(x) -1/2*sum((cholCov*x).^2);
gradLogPDF = @(x) -cov*x;

%% Case 1 - unit mass

sd = randi(10000000);
M = eye(2, 2);

% Standard sampler
rng(sd, 'twister');
s1 = hmcsample(startpoint, logPDF, gradLogPDF, 'NSamples', 100, 'Mass', M, 'PropSteps', 5);

% Riemann sampler
rng(sd, 'twister');
s2 = hmcsample(startpoint, logPDF, gradLogPDF, 'NSamples', 100, 'Metric', @(x) M, 'PropSteps', 5);

% Assert equal
assert(all(abs(s1(:) - s2(:)) < 1e-3));

%% Case 2 - lopsided mass

sd = randi(10000000);
M = eye(2, 2);
M(2,2) = 2;

% Standard sampler
rng(sd, 'twister');
s3 = hmcsample(startpoint, logPDF, gradLogPDF, 'NSamples', 100, 'Mass', M, 'PropSteps', 5);

% Riemann sampler
rng(sd, 'twister');
s4 = hmcsample(startpoint, logPDF, gradLogPDF, 'NSamples', 100, 'Metric', @(x) M, 'PropSteps', 5);

% Assert equal
assert(all(abs(s3(:) - s4(:)) < 1e-3));