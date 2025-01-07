% Creating Figure 3.5 on pg 74 of Bayesian Models of Perception and Action
close all

% Mean and standard deviation of the prior
mu_prior = 0;
sigma_prior = 3;

% Mean and standard deviation of the likelihood
mu_likelihood = 3;
sigma_likelihood = 2;

% Range of x values
x = linspace(-10, 10, 100);

% Define the prior and likelihood using normpdf
prior = normpdf(x, mu_prior, sigma_prior);
likelihood = normpdf(x, mu_likelihood, sigma_likelihood);

% Determine the posterior mean and standard deviation
% For Gaussian normal priors and likelihoods only
mu_posterior = ((mu_likelihood/(sigma_likelihood)^2) + (mu_prior/(sigma_prior)^2))/((1/(sigma_likelihood)^2) + (1/(sigma_prior)^2));
sigma_posterior = sqrt(1/(1/((sigma_likelihood)^2) + (1/(sigma_prior)^2)));

% Define the posterior using normpdf
posterior = normpdf(x, mu_posterior, sigma_posterior);

% Plot the prior, likelihood, and posterior
figure;
hold on;
plot(x, prior, 'k', 'LineWidth', 2);
plot(x, likelihood, 'r', 'LineWidth', 2);
plot(x, posterior, 'b', 'LineWidth', 2);
legend('Prior', 'Likelihood', 'Posterior');
xlabel('Hypothesized stimulus');
ylabel('Probability or likelihood');
hold off;
