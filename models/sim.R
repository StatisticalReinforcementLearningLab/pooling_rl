# rm(list = ls())
require(Matrix)
require(fda)

pZ <- 2

#### INPUT LIST ####
input = list()
# clipping
input$pi_max <- 0.8;
input$pi_min <- 0.1;

# prior and model spec
input$feat0 = function(z, x) c(1, z, x)
input$feat1 = function(z, x) c(1, z, x)
input$feat2 = function(z, x) c(1, z[1], x)
input$sigma <- 1
input$mu0 = rep(0, length(input$feat0(z = rep(0, pZ), x = 0)))
input$mu1 = rep(0, length(input$feat1(z = rep(0, pZ), x = 0)))
input$mu2 = rep(0, length(input$feat2(z = rep(0, pZ), x = 0)))
input$Sigma0 <- diag(10, length(input$feat0(z = rep(0, pZ), x = 0)));
input$Sigma1 <- diag(10, length(input$feat1(z = rep(0, pZ), x = 0)));
input$Sigma2 <- diag(10, length(input$feat2(z = rep(0, pZ), x = 0)));

# dosage and proxy spec
input$xi <- 10
input$eta.init = function(x) 0
input$gamma.mdp <- 0.9
input$lambda <- 0.9
input$p.sed <- 0.3
input$weight <- 0.5

# index in the batch set: (time, state = (Z, X, I), action, prob, reward)
input$z.index <- 2:(1+pZ)
input$x.index <- 2+pZ
input$avail.index <- 3+pZ 
input$action.index <- 4+pZ
input$prob.index <- 5+pZ 
input$rwrd.index <- 6+pZ

#### ALGORITHM #####

# action selection
prob.cal = function(z, x, mu, Sigma, eta, input){
  
  # immediate effect
  pos.mean <- c(input$feat2(z, x) %*% mu)
  pos.var <- c(t(input$feat2(z, x)) %*% Sigma %*% input$feat2(z, x))
  pos.var <- max(0, pos.var) # stability
  
  # delay effect
  margin <- eta (x) * input$xi
  
  # probability
  pit0 <- pnorm((pos.mean-margin)/sqrt(pos.var))
  
  # clipping
  prob =  min(c(input$pi_max, max(c(input$pi_min, pit0))))
  
  return(prob)
}

# policy update
policy.update = function(batch, input, proxy = T){
  
  post.cal = function(X, Y, sigma, mu, Sigma){
    
    inv.Sigma <- solve(Sigma)
    pos.mean <- c(solve(t(X)%*%X + sigma^2*inv.Sigma, t(X)%*%Y + sigma^2*inv.Sigma %*% mu));
    pos.var <- sigma^2 * solve(t(X) %*% X +  sigma^2 * inv.Sigma);
    
    list(mean = pos.mean, var = pos.var)
  }
  
  txt.eff.update = function(){
    
    # available subset
    avail <- batch[, input$avail.index]
    index <- (avail == 1)
    
    if(sum(index) == 0){
      
      # return the prior
      
      list(mean = input$mu2, var = input$Sigma2)
      
      
    }else{
      
      xz = matrix(batch[index, c(input$x.index, input$z.index)], nrow = sum(index))
      action <- batch[index, input$action.index]
      prob <- batch[index, input$prob.index]
      
      # forming the prior 
      mu.tmp <- c(input$mu1, input$mu2, input$mu2)
      Sigma.tmp <- as.matrix(bdiag(input$Sigma1, input$Sigma2, input$Sigma2))
      
      # feature matrix
      F1 <- t(apply(xz, 1, function(s) input$feat1(x = s[1], z = s[-1])))
      F2 <- t(apply(xz, 1, function(s) input$feat2(x = s[1], z = s[-1])))
      
      # calculate posterior (batch update)
      X.trn <- cbind(F1, prob * F2, (action-prob) * F2)
      Y.trn <- batch[index, input$rwrd.index]
      temp <- post.cal(X.trn, Y.trn, input$sigma, mu.tmp, Sigma.tmp)
      
      # return the post dist of txt eff
      txt.index <- tail(1:ncol(X.trn), ncol(F2)) # interaction terms
      list(mean = temp$mean[txt.index], var = temp$var[txt.index, txt.index])
      
    }
    
    
    
  }
  
  main.eff.update = function(){
    
    # available, no txt, subset
    avail <- batch[, input$avail.index]
    action <- batch[, input$action.index]
    index <- (avail == 1) & (action == 0)
    
    if(sum(index) == 0){
      
      # return the prior
      
      list(mean = input$mu1, var = input$Sigma1)
      
    }else{
      
      xz = matrix(batch[index, c(input$x.index, input$z.index)], nrow = sum(index))
      
      # forming the prior 
      mu.tmp <- input$mu1
      Sigma.tmp <- input$Sigma1
      
      # calculate posterior (batch update)
      X.trn <- t(apply(xz, 1, function(s) input$feat1(x = s[1], z = s[-1])))
      Y.trn <- batch[index, input$rwrd.index]
      temp <- post.cal(X.trn, Y.trn, input$sigma, mu.tmp, Sigma.tmp)
      
      # return the post dist of mian eff
      list(mean = temp$mean, var = temp$var)
      
    }
    
    
    
  }
  
  unavail.update = function(){
    
    # available, no txt, subset
    avail <- batch[, input$avail.index]
    index <- (avail == 0)
    
    if(sum(index) == 0){
      
      # return the prior
      
      list(mean = input$mu0, var = input$Sigma0)
      
    }else{
      
      xz = matrix(batch[index, c(input$x.index, input$z.index)], nrow = sum(index))
      
      # forming the prior 
      mu.tmp <- input$mu0
      Sigma.tmp <- input$Sigma0
      
      
      # calculate posterior (batch update)
      X.trn <- t(apply(xz, 1, function(s) input$feat1(x = s[1], z = s[-1])))
      Y.trn <- batch[index, input$rwrd.index]
      temp <- post.cal(X.trn, Y.trn, input$sigma, mu.tmp, Sigma.tmp)
      
      # return the post dist of mian eff
      list(mean = temp$mean, var = temp$var)
      
    }
    
  }
  
  # posterior update
  txt.est = txt.eff.update()
  
  
  if(proxy){
    
    # proxy update
    
    alpha0 <- unavail.update()$mean
    alpha1 <- main.eff.update()$mean
    alpha2 <- txt.est$mean
    
    p.avail <- mean(batch[, input$avail.index])
    X.null <- seq(0, 1/(1-input$lambda), by = 0.01)
    Z.trn <- batch[, input$z.index]
    
    # F0 <- t(sapply(X.null, function(x) apply(apply(Z.trn, 1, function(z) input$feat0(z, x)), 1, mean)))
    # F1 <- t(sapply(X.null, function(x) apply(apply(Z.trn, 1, function(z) input$feat1(z, x)), 1, mean)))
    # F2 <- t(sapply(X.null, function(x) apply(apply(Z.trn, 1, function(z) input$feat2(z, x)), 1, mean)))
    
    F.all <- t(sapply(X.null, function(x) apply(apply(Z.trn, 1, function(z) c(input$feat0(z, x), input$feat1(z, x), input$feat2(z, x))), 1, mean)))
    index0 <- 1:length(alpha0)
    index1 <- length(alpha0) + 1:length(alpha1)
    index2 <- tail(1:ncol(F.all), length(alpha2))
    F0 <- F.all[, index0]
    F1 <- F.all[, index1]
    F2 <- F.all[, index2]
    
    
    r0.vec = c(F0 %*% alpha0)
    r1.vec = c(F1 %*% alpha1)
    r2.vec = r1.vec + c(F2 %*% alpha2) 
    
    bsb <- create.bspline.basis (range=c(0, 1/(1-input$lambda)), nbasis=50, norder = 4)
    psi = function(x) c(eval.basis(x, bsb))
    
    psi.mat <- t(sapply(X.null, function(x) psi(x)))
    inv.cov <- solve(t(psi.mat) %*% psi.mat)
    
    psi.mat.irs = t(sapply(X.null, function(x) psi(input$lambda * x + 1)))
    psi.mat.drs = t(sapply(X.null, function(x) psi(input$lambda * x)))
    psi.mat.bar <- input$p.sed * psi.mat.irs  + (1-input$p.sed) * psi.mat.drs
    
    
    kmax <- 100;
    kk <- 1
    
    theta1 = rep(0, length(psi(0)));
    theta0 = rep(0, length(psi(0)));
    theta.bar = theta1 * p.avail + (1-p.avail) * theta0
    
    Y1.0 <- r1.vec + input$gamma.mdp * psi.mat.bar %*% theta.bar
    Y1.1 <- r2.vec + input$gamma.mdp * psi.mat.irs %*% theta.bar
    index <- (Y1.1 - Y1.0 > 0)
    Y1 <- Y1.0
    Y1[index] <- Y1.1[index]
    Y0 <- r0.vec + input$gamma.mdp * psi.mat.bar %*% theta.bar
    delta <- max(abs(Y1 - psi.mat%*% theta1), abs(Y0 - psi.mat %*% theta0))
    
    
    delta.thres <- 1e-2;
    while(kk < kmax & delta > delta.thres){
      
      # new theta
      theta1 <- inv.cov %*% t(psi.mat) %*% Y1
      theta0 <- inv.cov %*% t(psi.mat) %*% Y0
      
      theta.bar = theta1 * p.avail + (1-p.avail) * theta0
      
      # Bellman operator
      Y1.0 <- r1.vec + input$gamma.mdp * psi.mat.bar %*% theta.bar
      Y1.1 <- r2.vec + input$gamma.mdp * psi.mat.irs %*% theta.bar
      index <- (Y1.1 - Y1.0 > 0)
      Y1 <- Y1.0
      Y1[index] <- Y1.1[index]
      Y0 <- r0.vec + input$gamma.mdp * psi.mat.bar %*% theta.bar
      
      delta <- max(abs(Y1 - psi.mat%*% theta1), abs(Y0 - psi.mat %*% theta0))
      kk <- kk + 1
    
      #cat(kk, delta, "\n")
    }
    
    if(kk == kmax){
      warning("Not Converge")
    }
    
    
    eta.fn = function(x) {
      
      eta.hat <- c((1-input$p.sed)* t(theta.bar)%*%(psi(input$lambda*x)-psi(input$lambda*x+1)) * (1-input$gamma.mdp))
      input$weight * eta.hat + (1-input$weight) * input$eta.init(x)
      
      
    }
    
    list(eta = eta.fn, mean = txt.est$mean, var = txt.est$var)
    
  }else{
    
    list(mean = txt.est$mean, var = txt.est$var)
    
  }
  
  
  
}

#### SIMULATION ####
sim = function(){
  
  # simulation
  nT <- 450
  
  # initial state
  Z.init <- runif(pZ)
  X.init <- 0;
  I.init <- (runif(1) < 0.8)
  
  # dosage transition
  gen_nextdosage = function(x, a) {
    
    anti.sed <- as.numeric(runif(1) < 0.3)
    event <- (anti.sed+a)>0
    x.next <- 0.9*x + event
    
    return(x.next)
  }
  
  
  # policy initilization
  mu.beta <- input$mu2
  Sigma.beta <- input$Sigma2;
  eta.fn <- input$eta.init
  
  # placeholder
  batch <- NULL
  
  # performance
  tot.rwrd <- 0
  
  for(t in 1:nT){
    
    
    # current state 
    
    if(t==1){
      
      Z.next <- Z.init
      X.next <- X.init;
      I.next <- I.init;
      
    }
    
    Z <- Z.next
    X <- X.next
    I <- I.next
    
    # action selection
    
    if(I == 1){
      
      # calculate prob
      
      prob <- prob.cal(Z, X, mu.beta, Sigma.beta, eta.fn, input)
      
      # sample the action
      A <- as.numeric(runif(1) < prob)
      
    }else{
      
      prob = 0
      A = 0
      
    }
    
    # Collect Reward and State Transition
    R <-  1 + Z[1] + Z[1]^2 - 0.05 * X + A * (1 - 0.1*X) + rnorm(1, sd = 1)
    Z.next <- runif(pZ)
    X.next <- gen_nextdosage(X, A)
    I.next <- (runif(1) < 0.8)
    
    
    # add to the batch set
    batch <- rbind(batch, c(t, Z, X, I, A, prob, R))
    
    # performance 
    tot.rwrd <- tot.rwrd + R
    
    
    # nightly update
    
    if(t%%5 == 0){
      
      if(t %% 35 == 0){
        
        # weekly update the proxy value (only for sim)
        
        temp = policy.update(batch, input, proxy = T)
        mu.beta = temp$mean
        Sigma.beta = temp$var
        eta.fn <- temp$eta
        
      }else{
        
        # only update the txt effect posterior
        
        temp = policy.update(batch, input, proxy = F)
        mu.beta = temp$mean
        Sigma.beta = temp$var
        
      }
      
      
      #print(t%/%5)
      
    }
    
    
  }
  
  return(tot.rwrd)
  
}


#### STANDARD TS Bandit ####


# action selection
prob.cal.ts = function(z, x, mu, Sigma, input){
  
  # immediate effect
  pos.mean <- c(input$feat2(z, x) %*% mu)
  pos.var <- c(t(input$feat2(z, x)) %*% Sigma %*% input$feat2(z, x))
  print(pos.var)
  pos.var <- max(0, pos.var) # stability
  
  # probability
  pit0 <- pnorm((pos.mean)/sqrt(pos.var))
  
  # clipping
  prob =  min(c(input$pi_max, max(c(input$pi_min, pit0))))
  
  return(prob)
}

# policy update
policy.update.ts = function(batch, input){
  print(batch) 
  post.cal = function(X, Y, sigma, mu, Sigma){
    
    inv.Sigma <- solve(Sigma)
    pos.mean <- c(solve(t(X)%*%X + sigma^2*inv.Sigma, t(X)%*%Y + sigma^2*inv.Sigma %*% mu));
    pos.var <- sigma^2 * solve(t(X) %*% X +  sigma^2 * inv.Sigma);
    
    list(mean = pos.mean, var = pos.var)
  }
  
  txt.eff.update = function(){
    
    # available subset
    avail <- batch[, input$avail.index]
    index <- (avail == 1)
    
    if(sum(index) == 0){
      
      # return the prior
      
      list(mean = input$mu2, var = input$Sigma2)
      
      
    }else{
      
      xz = matrix(batch[index, c(input$x.index, input$z.index)], nrow = sum(index))
      action <- batch[index, input$action.index]
      prob <- batch[index, input$prob.index]
      
      # forming the prior 
      mu.tmp <- c(input$mu1, input$mu2)
      Sigma.tmp <- as.matrix(bdiag(input$Sigma1, input$Sigma2))
      
      # feature matrix
      F1 <- t(apply(xz, 1, function(s) input$feat1(x = s[1], z = s[-1])))
      F2 <- t(apply(xz, 1, function(s) input$feat2(x = s[1], z = s[-1])))
      
      # calculate posterior (batch update)
      X.trn <- cbind(F1, action * F2)
      Y.trn <- batch[index, input$rwrd.index]
      temp <- post.cal(X.trn, Y.trn, input$sigma, mu.tmp, Sigma.tmp)
      
      # return the post dist of txt eff
      txt.index <- tail(1:ncol(X.trn), ncol(F2)) # interaction terms
      list(mean = temp$mean[txt.index], var = temp$var[txt.index, txt.index])
      
    }
    
    
    
  }
  
  # posterior update
  txt.est = txt.eff.update()
  
  
  list(mean = txt.est$mean, var = txt.est$var)
  
  
  
}

# simulation
sim.ts = function(){
  
  # simulation
  nT <- 450
  
  # initial state
  Z.init <- runif(pZ)
  X.init <- 0;
  I.init <- (runif(1) < 0.8)
  
  # dosage transition
  gen_nextdosage = function(x, a) {
    
    anti.sed <- as.numeric(runif(1) < 0.3)
    event <- (anti.sed+a)>0
    #print('event')
    #print(event)
    x.next <- 0.9*x + event
    
    return(x.next)
  }
  
  
  # policy initilization
  mu.beta <- input$mu2
  Sigma.beta <- input$Sigma2;
  
  # placeholder
  batch <- NULL
  
  
  # performance matrix
  tot.rwrd <- 0
  
  
  for(t in 1:nT){
    
    
    # current state 
    
    if(t==1){
      
      Z.next <- Z.init
      X.next <- X.init;
      I.next <- I.init;
      
    }
    
    Z <- Z.next
    X <- X.next
    I <- I.next
    
    # action selection
    
    if(I == 1){
      
      # calculate prob
      
      prob <- prob.cal.ts(Z, X, mu.beta, Sigma.beta, input)
      
      # sample the action
      A <- as.numeric(runif(1) < prob)
      
    }else{
      
      prob = 0
      A = 0
      
    }
    
    # Collect Reward and State Transition
    R <-  1 + Z[1] + Z[1]^2 - 0.05 * X + A * (1 - 0.1*X) + rnorm(1, sd = 1)
    Z.next <- runif(pZ)
    X.next <- gen_nextdosage(X, A)
    I.next <- (runif(1) < 0.8)
    
    
    # add to the batch set
    batch <- rbind(batch, c(t, Z, X, I, A, prob, R))
    
    
    # performance
    tot.rwrd <- tot.rwrd + R
    
    # nightly update
    
    if(t%%5 == 0){
      
       #update the txt effect posterior
      temp = policy.update.ts(batch, input)
      print(temp)
      mu.beta = temp$mean
      Sigma.beta = temp$var
      
      
      #print(t%/%5)
      
    }
    
    
  }
  print(tot.rwrd)
  return(batch)
  
}





#### Sample Run ####
sim()
sim.ts()



