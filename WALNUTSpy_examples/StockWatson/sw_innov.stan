
data {
  int<lower=0> T;
  array[T] real y;
}

parameters {
  real tSigma;
  real z1;
  array[T-2] real zinn;
  real x1;
  array[T-1] real xinn;
  real tau1;
  array[T-1] real tauinn;
}


transformed parameters {
  real sigma;
  real gpri;
  array[T-1] real z;
  array[T] real x;
  array[T] real tau;
  
  sigma = exp(-0.5*tSigma);
  gpri = exp(tSigma);
  
  z[1]=z1;
  for(t in 2:(T-1)) z[t] = z[t-1] + sigma*zinn[t-1];
  
  x[1]= x1;
  for(t in 2:T) x[t] = x[t-1] + sigma*xinn[t-1];
  
  tau[1] = tau1;
  for(t in 2:T) tau[t] = tau[t-1] + exp(0.5*z[t-1])*tauinn[t-1];
  
  
}

model {
  target+= 5.0*tSigma - 0.5*exp(tSigma); 
  //z1 ~ normal(0.0, 1.0); // to be removed
  //x1 ~ normal(0.0, 1.0); // to be removed
  //tau1 ~ normal(0.0, 1.0); // to be removed
  zinn ~ normal(0.0,1.0);
  xinn ~ normal(0.0,1.0);
  tauinn ~ normal(0.0,1.0);
  for(t in 1:T){
    y[t] ~ normal(tau[t], exp(0.5*x[t]));
  }
  
}

