% simulation of the new efficient Bayesian estimator
% see: wei/stocker, natneuro, 2015
%
% magnitude space
%
% stimulus noise: Gaussian
% sensory(internal) noise: Gaussian
%
% modified from code by astocker; 10.24.2016

clear
%close all

%% PARMS

% prior exponent
k = -1;

% Low frequency roll-off (Hz) of the prior
rolloffHz = 0;

% stimulus noise - Gaussian
stim_noise = false; % flag;
sigma_stim = 0.15;

% sensory noise - Gaussian
sigma_sen = 0.01;

% Flicker frequency range
xRangeLinear = [8 200];

% The number of samples along the stimulus range
nSamples = 1000;

% Weird things happen on the unbounded model; censor some of these points
% from plotting
nCensor = 50;

% The example stimulus index to show in the plot
exampleIdx = 400;

% Sampling parms. We operate on the log-transformed stimulus values
x0 = linspace(log10(xRangeLinear(1)),log10(xRangeLinear(2)),nSamples);  % stimulus domain
d0 = x0(2)-x0(1);

x = linspace(log10(xRangeLinear(1)),log10(xRangeLinear(2)),nSamples+1); % shifted to be made non-overlapping?
dx = x(2)-x(1);
x(1) = [];
x = x - dx/2;

% variable and measurment
th = x';  % column
m = x;  % row

%% generative model
% prior
% 1/f^2 distribution of temporal frequency
%
myFun = @(x) (10.^x).^k;
p = myFun(x0);
rolloffIdx = find(10.^x>=rolloffHz,1);
p(1:rolloffIdx) = p(rolloffIdx);

p = p./trapz(x0,p);
p_th = interp1(x0,p,th); 

% efficient coding - mapping to sensory space (tilde, t)
x0t = range(x0)*cumtrapz(x0,p)+min(x0); 
tht = interp1(x0,x0t,th); % column
mt = interp1(x0,x0t,m); % row

% jacobian for prob transform into sensory space
%% GKA GUESS AS TO HOW TO MAP INTO SENSORY SPACE
a = 1./(range(x0)*p_th+min(x0));

if stim_noise
    % stimulus noise model - normpdf(x,mu,kappa)
    p_m_gv_th = normpdf(repmat(m,[nSamples 1]),repmat(th,[1 nSamples]),sigma_stim);

    % prob transform to internal space
    p_mt_gv_tht = repmat(a',[nSamples 1]) .* p_m_gv_th; 
    
    % plus sensory noise - convolution with normpdf
    % needs to be on regular grid -> resample along m
    % attention: interp1 is defined on column vector with array; therefore
    % the necessity to transpose
    q = interp1(mt',p_mt_gv_tht',x','linear','extrap')';

    % convolution
    b = normpdf(repmat(x',[1 nSamples]),repmat(x,[nSamples 1]),sigma_sen);  
    p_mt_gv_tht = dx * q * b';

    % resample back
    p_mt_gv_tht = interp1(x',p_mt_gv_tht',mt','linear','extrap')';

else % only sensory noise

    p_mt_gv_tht = normpdf(repmat(mt,[nSamples 1]),repmat(tht,[1 nSamples]),sigma_sen);

end

% transform conditional distribution back to stimulus space 
p_m_gv_th = p_mt_gv_tht./repmat(a',nSamples,1); % row -> noise distribution


%% inference

% in stimulus space because loss-function is defined in stimulus space
p_th_gv_m = p_m_gv_th.*repmat(p_th,1,nSamples); % column -> likelihood
p_th_gv_m = p_th_gv_m./repmat(trapz(th,p_th_gv_m),[nSamples 1]);

% BLS estimator - mean of posterior.
%% GKA GUESS AS TO HOW TO MAP POSTERIOR TO THETA DOMAIN
thh_gv_m = range(x0)*mean(p_th_gv_m .* repmat(th,1,nSamples),1)+min(x0);

% distribution of estimates
% marginalization over p_m_gv_th (variable exchange)
c = 1./gradient(thh_gv_m,dx); % jacobian
p_thh_gv_th = repmat(c,nSamples,1) .* p_m_gv_th;

% resampling along thh axis
%% Had to comment this out; not sure how it is supposed to operate
%p_thh_gv_th = interp1(thh_gv_m',p_thh_gv_th',th,'linear','extrap')';
    
% expectation and bias and variance
%% GKA GUESS AS TO HOW TO MAP EXPECTATION TO THETA DOMAIN
E_thh_gv_th = range(x0)*mean(p_thh_gv_th.* repmat(th',nSamples,1),2)+min(x0);
bias = E_thh_gv_th-th;


%% GKA Not sure how to calculate the variance. Making a dummy variable for now
var = [];

% max
[Imax, Jmax] = find(p_thh_gv_th==max(p_thh_gv_th));

%% plots
fh = figure;
fh.Position = [100 100 300 900];

% prior and bias
subplot(3,1,1);
plot(th,zeros(size(th)),':k'); % zero
hold on;
plot(th,p,'b-'); % prior
plot(th(1+nCensor:end-nCensor),bias(1+nCensor:end-nCensor),'r-'); % bias
%plot(th,var,'m-'); % variance
%plot([-pi pi],[0 0],'k--');
%axis([-pi pi ylim]);
legend({'','prior','bias'});
xlabel('stimulus variable [log Hz]');
ylabel('bias | prior');
title('prior, bias, and variance');
xlim(log10(xRangeLinear));

% some example dstribution p_thh_gv_th (@ th(600))
subplot(3,1,2);
plot(th,p_thh_gv_th(exampleIdx,:),'b-'); % distribution of estimates for example value
hold on;
plot(E_thh_gv_th(exampleIdx),0,'bx','MarkerSize',15); % expected stimulus estimate
plot(th(exampleIdx),0,'go','MarkerSize',15); % true stimulus value
plot(th(Imax(exampleIdx)),0,'r.','MarkerSize',15); % max
xlabel('stimulus variable [log Hz]');
ylabel('prob density');
legend({'estimate distribution','mean','true','max'});
title('estimate distribution for example stimulus frequency');
xlim(log10(xRangeLinear));

% full distribution
subplot(3,1,3);
colormap('hot');
contour(p_thh_gv_th,20);
axis square;
grid on;
ticks = [1 nSamples/2 nSamples];
xticks(ticks);
xticklabels(num2str(round(th(ticks),1)));
xlabel('stimulus variable [log Hz]');
yticks(ticks);
yticklabels(num2str(round(th(ticks),1)));
ylabel('estimate [log Hz]');
title('prob. density of estimates','FontWeight','normal');
