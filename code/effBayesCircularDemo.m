% simulation of the new efficient Bayesian estimator
% see: wei/stocker, natneuro, 2015
%
% cicular space
%
% stimulus noise: vonMises
% sensory(internal) noise: vonMises
%
% astocker
% 10.24.2016
%
% rev history
% 1. only internal (sensory) noise
% 2. stimulus noise added



%% PARMS
% stimulus noise - vonMises
stim_noise = 1; % flag; set this to 0 for kappa1>600;
k_stim = 10; 

% sensory noise - vonMises
k_sen = 5;

% sampling parms
n = 1000;
x0 = linspace(-pi,pi,n);  % end to end
d0 = x0(2)-x0(1);

x = linspace(-pi,pi,n+1); % shifted by 1/2 - non-overlapping for marginalization
dx = x(2)-x(1);
x(1) = [];
x = x - dx/2;

% variable and measurment
th = x';  % column
m = x;  % row

%% generative model
% prior
% example here: approximate prior for visual orientation (natural image statistics)
% replace with any other prior density
cp = 0.8;
p = cp*((1/cp)-abs(sin(x0).^1));
p = p./(trapz(x0,p)); 
p_th = interp1(x0,p,th); 

% efficient coding - mapping to sensory space (tilde, t)
x0t = (2*pi*cumtrapz(x0,p))-pi; 
tht = interp1(x0,x0t,th); % column
mt = interp1(x0,x0t,m); % row

% jacobian for prob transform into sensory space
a = 1./(2*pi*p_th);

if stim_noise
    % stimulus noise model - vonMises(x,mu,kappa)
    p_m_gv_th = vonMises(repmat(m,[n 1]),repmat(th,[1 n]),k_stim);

    % prob transform to internal space 
    p_mt_gv_tht = repmat(a',[n 1]) .* p_m_gv_th; 
    
    % plus sensory noise - convolution with vonMises
    % needs to be on regular grid -> resample along m
    % attention: interp1 is defined on column vector with array; therefore
    % the necessity to transpose
    q = interp1(mt',p_mt_gv_tht',x','linear','extrap')';

    % convolution
    b = vonMises(repmat(x',[1 n]),repmat(x,[n 1]),k_sen);  
    p_mt_gv_tht = dx * q * b';

    % resample back
    p_mt_gv_tht = interp1(x',p_mt_gv_tht',mt','linear','extrap')';
else % only sensory noise
    p_mt_gv_tht = vonMises(repmat(mt,[n 1]),repmat(tht,[1 n]),k_sen);
end

% transform conditional distribution back to stimulus space 
p_m_gv_th = p_mt_gv_tht./repmat(a',n,1); % row -> noise distribution



%% inference
% in stimulus space because loss-function is defined in stimulus space

p_th_gv_m = p_m_gv_th.*repmat(p_th,1,n); % column -> likelihood
p_th_gv_m = p_th_gv_m./repmat(trapz(th,p_th_gv_m),[n 1]);

% BLS estimator - mean of posterior
thh_gv_m = atan2(sum(p_th_gv_m .* repmat(sin(th),1,n),1), sum(p_th_gv_m .* repmat(cos(th),1,n),1));

% distribution of estimates
% marginalization over p_m_gv_th (variable exchange)
c = 1./gradient(thh_gv_m,dx); % jacobian
p_thh_gv_th = repmat(c,n,1) .* p_m_gv_th;

% resampling along thh axis
p_thh_gv_th = interp1(thh_gv_m',p_thh_gv_th',th,'linear','extrap')';
    
% expectation and bias and variance
a = trapz(th,p_thh_gv_th .* repmat(sin(th'),n,1),2);
b = trapz(th,p_thh_gv_th .* repmat(cos(th'),n,1),2);

% E_thh_gv_th = atan2(sum(p_thh_gv_th.* repmat(sin(th'),n,1),2), sum(p_thh_gv_th .* repmat(cos(th'),n,1),2));
E_thh_gv_th = atan2(a,b);

bias = E_thh_gv_th-th;
var = 1 - sqrt(a.^2+b.^2); % circular
std = sqrt(2*(1 - sqrt(a.^2+b.^2))); % circular


% max
[Imax, Jmax] = find(p_thh_gv_th==max(p_thh_gv_th));

%% plots
figure(1);
clf;
% prior and bias
subplot(3,1,1);
plot(th,p,'b-'); % prior
hold on;
plot(th,bias,'r-'); % bias
plot(th,var,'m-'); % circular variance
plot([-pi pi],[0 0],'k--');
axis([-pi pi ylim]);
xlabel('stimulus variable (rad)');
ylabel('bias (rad) / prior prob');
title('prior, bias, and variance');

% some example dstribution p_thh_gv_th (@ th(600))
subplot(3,1,2);
ind= 600;
plot(th,p_thh_gv_th(ind,:),'b-'); % distribution of estimates for example value
hold on;
plot(E_thh_gv_th(ind),0,'b.'); % expected stimulus estimate
plot(th(ind),0,'g.'); % true stimulus value
plot(th(Imax(ind)),0,'r.'); % max
axis([-pi pi ylim]);
xlabel('stimulus variable (rad)');
ylabel('prob');
legend({'estimate distribution','mean','true','max'});
title('estimate distribution for example stimulus orientation');


% full distribution
subplot(3,1,3);
colormap('hot');
contour(p_thh_gv_th,20);
axis square;
grid on;
ticks = [1 n/2 n];
xticks(ticks);
xticklabels(num2str(round(th(ticks),1)));
xlabel('stimulus variable (rad)');
yticks(ticks);
yticklabels(num2str(round(th(ticks),1)));
ylabel('estimate (rad)');
title('prob. density of estimates','FontWeight','normal');

%% ROUTINES

% VONMISESPDF
function d = vonMises(x,mu,kappa)
% Usage: d = vonMises(x,mu,kappa)
% If kappa is very large (and the distribution very narrow),
% we can't use the formula below to compute von mises pdf, 
% as it just gives NANs.  But when the distribution is very narrow,
% it approaches a Gaussian.

if kappa>600
  x = x - mu;
  x = wrapToPi(x);
  d = normpdf(x,0,1./(sqrt(kappa)));
else
  d = exp(kappa.*cos((x-mu))) ./ (2*pi*besseli(0,kappa));
end
end
