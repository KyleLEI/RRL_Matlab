function [ J, grad ] = costFunction( Xn, X, theta)
%COSTFUNCTION Summary of this function goes here
%   Detailed explanation goes here
    miu = 1;
    delta = 0.001;
        
    d = size(X,2);
    M= (length(theta)-2)/d;
    T = length(X) - M;
    
    syms a
    syms b
    dSdA = diff(a/sqrt(b-a^2),a);
    dSdB = diff(a/sqrt(b-a^2),b);
   


    Ft = updateFt(Xn, theta, T);     
    
    [Ret, sharp] = rewardFunction(X, miu, delta, Ft, M);
    J = sharp * -1;


    dFt = zeros(d*M+2,T+1);
    for i = 2:T+1,
        xt = [1; reshape(Xn(i-1:i+M-2,:),[],1); Ft(i-1)];
        dFt(:,i) = (1 - tanh(theta' * xt) ^ 2) * (xt + theta(d*M+2,1)*dFt(:,i-1));
    end
    
    
    dRtFt = -1 * miu * delta * sign(Ft(2:end)-Ft(1:T));
    dRtFtt = miu * reshape(X(M+1:T+M),[],1) + miu * delta * sign(Ft(2:end)-Ft(1:T));
    A = sum(Ret) / T;
    B = sum(Ret.*Ret) / T;

    prefix = repmat(subs(subs(dSdA,a,A),b,B), T, 1)/T + subs(subs(dSdB,a,A),b,B)*2*Ret/T;
    %prefix = repmat((1/(- A^2 + B)^(1/2) + A^2/(B - A^2)^(3/2))/M, M, 1) + (-A/(2*(B - A^2)^(3/2))) * 2 * Ret / M
    %prefix = ones(M,1)
    
    grad = sum(repmat(prefix', d*M+2, 1) .* (repmat(dRtFt', d*M+2, 1) .* dFt(:,2:end) + repmat(dRtFtt', d*M+2, 1) .* dFt(:,1:T)), 2);
    grad = grad * -1;

    grad = double(grad);
    J = double(J);
end

