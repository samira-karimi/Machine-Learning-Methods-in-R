load("gh.rda")
dim(gh.train$x)
dim(gh.test$x)

library(foreach)
library(glmnet)

# p>n then lasso is work better.

###Elastic Net ###

fit.enet0 <- glmnet(gh.train$x, gh.train$y,alpha=.5)
plot(fit.enet0)

# 10 fold croos validation
fit.enet <- cv.glmnet(gh.train$x, gh.train$y, type.measure = "mse",alpha=.5)
plot(fit.enet)

# Best lambda
bestlam_enet = fit.enet$lambda.min
bestlam_enet

# Predict the glmnet and plot the Coef for ElasticNet
enet.coef <- predict(fit.enet0, type = "coefficient", s = bestlam_enet)
enet.coef[enet.coef !=0]
enet.coef[which(enet.coef!=0)]
plot(enet.coef[-1], col = "blue" , pch = 15)



### Lasso ###

fit.lasso0 <- glmnet(gh.train$x, gh.train$y,alpha=1)
plot(fit.lasso0)

# 10 fold cross validation
fit.lasso <- cv.glmnet(gh.train$x, gh.train$y, type.measure = "mse",alpha=1)
plot(fit.lasso)

# Best lambda
bestlam_lasso=fit.lasso$lambda.min
bestlam_lasso

# Predict the glmnet and plot the Coef for lasso
lasso.coef <- predict(fit.lasso0, type = "coefficient", s = bestlam_lasso)
lasso.coef[lasso.coef !=0]
lasso.coef[which(lasso.coef!=0)]
plot(lasso.coef[-1] , col = "red",pch = 15)



### Ridge ###

fit.ridge0 <- glmnet(gh.train$x, gh.train$y,alpha=0)
plot(fit.ridge0)

# 10 fold cross validation
fit.ridge <- cv.glmnet(gh.train$x, gh.train$y, type.measure = "mse",alpha=0)
plot(fit.ridge)

# Best lambda
bestlam_ridge=fit.ridge$lambda.min
bestlam_ridge

## Predict the glmnet and plot the Coef for Ridge
ridge.coef <- predict(fit.ridge0, type = "coefficient", s = bestlam_ridge)
ridge.coef[ridge.coef !=0]
ridge.coef[which(ridge.coef!=0)]
plot(ridge.coef[-1], col = "purple", pch = 15)


#2- Plot the cross validation error (cvm) from cv.glmnet

#plot the cvm for Elastic Net
plot(fit.enet$cvm)
#plot the cvm for Lasso
plot(fit.lasso$cvm)
#plot the cvm for Ridge
plot(fit.ridge$cvm)


#3- Predict yhat and Caculate the MSE

yhat.enet <- predict(fit.enet, s=fit.enet$lambda.1se, newx=gh.test$x)
(mse.enet <- mean(( gh.test$y - yhat.enet)^2))

yhat.lasso <- predict(fit.lasso, s=fit.lasso$lambda.1se, newx=gh.test$x)
(mse.enet <- mean(( gh.test$y - yhat.enet)^2))

yhat.ridge <- predict(fit.ridge, s=fit.ridge$lambda.1se, newx=gh.test$x)
(mse.ridge <- mean((gh.test$y - yhat.ridge)^2))


########################################################
#                   Decision tree
########################################################
# # Load CART packages
library(rpart)
# install rpart package
install.packages("rpart.plot")
library(rpart.plot)
# CART model
latlontree = rpart(MEDV ~ LAT + LON, data=boston)
# Plot the tree using prp command defined in rpart.plot package
prp(latlontree)

########################################################
#                    RandomForest
########################################################

library(randomForest)

fit = randomForest(gh.train$y ~ ., gh.train$x, ntree = 1000, mtry = 1000)
summary(fit)

#Predict Output

predicted_rtest = predict(fit, gh.test$x)
predicted_rtrain = predict (fit, gh.train$x)


mean((predicted_rtest - gh.test$y)^2)
plot(predicted_rtest, gh.test$y); abline(0,1)
mean((predicted_rtrain - gh.train$y)^2)
plot(predicted_rtrain, gh.train$y); abline(0,1)


######################################################
#                     NeuralNetwork
######################################################

install.packages("NeuralNetTools")
library(nnet)
library(NeuralNetTools)
library(MASS)


fit.NN <- nnet(gh.train$x, gh.train$y, size = 5, rang = 0.2, decay = 5e-4, maxit = 1200) 
predicted_ntest = predict(fit.NN, gh.test$x)
predicted_ntrain = predict (fit.NN, gh.train$x)

mean((predicted_ntest - gh.test$y)^2)
plot(predicted_ntest, gh.test$y); abline(0,1)


mean((predicted_ntrain - gh.train$y)^2)
plot(predicted_ntrain, gh.train$y); abline(0,1)




########################################################
#                        OlS
########################################################




train.y = as.matrix(gh.train$y)
test.y = as.matrix(gh.test$y)
y = rbind(train.y , test.y)
dim(y)

train.x = as.matrix(gh.train$x)
test.x = as.matrix(gh.test$x)
x = rbind(train.x , test.x)
dim(x)

mydata = cbind2(y, x)
mydata = data.frame(mydata)


OLSreg<- lsfit(gh.train$x,gh.train$y)

N = length(test.y)
XX = cbind(rep(1,N), test.x)

Yhat = XX %*% OLSreg$coef      

dof = N - length(OLSreg$coef)

errorvar = sum((OLSreg$res)^2) / (N - length(OLSreg$coef)) 

ols.pred.error<- mean((test.y-(Yhat))^2)
ols.pred.error
plot(test.y-(Yhat))

qqnorm(test.y-(Yhat))
qqline(test.y-(Yhat))


######################### FINISH ##########################


myTrainData = cbind2(train.y, train.x)
myTrainData = data.frame(myTrainData)

fit.OLS<-lm(train.y~., data = myTrainData)

TTT=data.frame(gh.test$x)

yhat.OLS=predict(fit.OLS, data.frame(XX))


mean((gh.test$y - yhat.OLS) ^ 2)

plot(gh.test$y-yhat.OLS)

newd <- data.frame(test.x)
predict(fitols, newd)

predict.lm(fitols,newdata = myTestData)

N = length(test.y)
XX = cbind(rep(1,N), test.x)

Yhat = XX %*% fit.OLS$coefficients      #gives you the prediction..

dof = N - length(fitols$coefficients)

errorvar = sum((fitols$res)^2) / (N - length(fitols$coefficients))

ols.pred.error<- mean((test.y-(Yhat))^2)
plot(test.y-(Yhat))
