clear;
close;
opts = detectImportOptions('covid_19_clean_complete.csv');
opts = setvartype(opts,{'Country_Region'},{'string'});

wti = readtimetable('wti.csv');
covid = readtimetable('covid_19_clean_complete.csv',opts);

covid.Date = covid.Date + calyears(2000); % because of 2-digit year
wti.Date = wti.Date + calyears(2000);

dates = intersect(covid.Date,wti.Date); % find the common timeframe
states = wti(dates,{'Open','Close'});
covid_confirmed = zeros(length(dates),1);
covid_deaths = zeros(length(dates),1);

% aggregate same-day COVID-19 figures
for i=1:length(dates)
    date = dates(i);
    covid_confirmed(i) = sum(table2array(covid(date,{'Confirmed'})));
    covid_deaths(i) = sum(table2array(covid(date,{'Deaths'})));
end

states = addvars(states,covid_confirmed,covid_deaths);


M = 10; 
T = 150; % The number of time series inputs to the trader, Apr 20 crash 136
N = 30;


price = [states.Open';states.Close'];
price = reshape(price,[],1);
price(1) = []; % assume the agent knows only the last day's infection figures
price_ret = (price(2:end)-price(1:end-1))./price(1:end-1);
covid_confirmed_rate = (covid_confirmed(2:end)-covid_confirmed(1:end-1))./covid_confirmed(1:end-1);
covid_confirmed_rate = reshape([covid_confirmed_rate';covid_confirmed_rate'],[],1); % duplicate
covid_deaths_rate = (covid_deaths(2:end)-covid_deaths(1:end-1))./covid_deaths(1:end-1);
covid_deaths_rate = reshape([covid_deaths_rate';covid_deaths_rate'],[],1);

% Without COVID-19 statistics
% initial_w = ones(M+2,1);
% X = price_ret;

% With COVID-19 statistics
initial_w = ones(3*M+2,1); % initialize theta
X = [price_ret covid_confirmed_rate covid_deaths_rate];

Xn = featureNormalize(X);

 %  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 30, 'PlotFcns', @optimplotfval);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[w, cost, EXITFLAG,OUTPUT] = fminunc(@(t)(costFunction(Xn(1:M+T,:), X(1:M+T,:), t)), initial_w, options)


Ft = updateFt(Xn(1:M+T,:), w, T);

miu = 1;
delta = 0.001;

[Ret, sharp] = rewardFunction(X(1:M+T,:), miu, delta, Ft, M);
Ret = Ret + 1;
%size(Ret), size(Ft)
for i = 2:length(Ret)
    Ret(i) = Ret(i-1)*Ret(i); 
end

figure;
subplot(4,1,1);
plot(price(M+2:M+T+2));
axis([0, T, min(price(M+2:M+T+2))*0.95, max(price(M+2:M+T+2))*1.05]);
title('WTI');
subplot(4,1,2);
hold on;
plot(covid_confirmed_rate,'DisplayName','Confirmed');
plot(covid_deaths_rate,'DisplayName','Deaths');
axis([0,T,0 0.5]);
legend('Location','northwest');
title('COVID-19 Rates');
subplot(4,1,3);
plot(Ft(2:end));
axis([0, length(Ft)-1, -1.05, 1.05]);
title('Trader Function F_t');
subplot(4,1,4);
plot(Ret(:));
axis([0, length(Ret), min(Ret), max(Ret)]);
title('Return');
 
pause;

 %Ft(T:T+N)
 
 pI = 1;
 Ret = zeros(pI*N, 1);
 %size(Ret)
 Ft = zeros(pI*N, 1);
 for i = 0:pI-1
     Ftt = updateFt(Xn(T+i*N:T+(i+1)*N+M,:), w, N);
     
     [Rett, sharp] = rewardFunction(X(T+i*N:T+(i+1)*N+M,:), miu, delta, Ftt, M);
     Rett = Rett + 1;
     Ret(i*N+1:(i+1)*N) = Rett;
     
     for j = i*N+1:(i+1)*N
         if j-1 ~= 0
             Ret(j) = Ret(j-1)*Ret(j); 
         end
     end
     
     Ft(i*N+1:(i+1)*N) = Ftt(2:end);
     
%      [w, cost, EXITFLAG,OUTPUT] = fminunc(@(t)(costFunction(Xn(i*N+1:i*N+M+T,:), X(i*N+1:i*N+M+T,:), t)), w, options);
 end
 
figure;
subplot(4,1,1);
D = price(M+T+3:M+T+2+pI*N);
plot(D);
axis([0, pI*N, min(D)*0.95, max(D)*1.05]);
title('WTI');
subplot(4,1,2);
C_c = covid_confirmed_rate(M+T+3:M+T+2+pI*N);
C_d = covid_deaths_rate(M+T+3:M+T+2+pI*N);
hold on;
plot(C_c,'DisplayName','Confirmed');
plot(C_d,'DisplayName','Deaths');
axis([0,pI*N,0 0.5]);
legend('Location','northwest');
title('COVID-19 Rates');
subplot(4,1,3);
plot(Ft(2:end));
axis([0, length(Ft)-1, -1.05, 1.05]);
title('Trader Function F_t');
subplot(4,1,4);
plot(Ret(:));
axis([0, length(Ret), min(Ret), max(Ret)]);
title('Return');