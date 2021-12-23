library(stats)
library(dplyr)
library(randomForest)
mydata<-read.csv("diabetes.csv")
str(mydata)
index<-sample(2,nrow(mydata),replace=TRUE, prob = c(0.7,0.3))

#training dat a. 
set.seed(123)
training <-mydata[index==1,]
testing<-mydata[index==2,]

str(mydata)
#random forest 

rfm = randomForest(as.factor(Outcome)~.,data=training)

Outcome_pred <-predict(rfm,testing)
testing$Outcome_pred=Outcome_pred

cfm<-table(testing$Outcome,testing$Outcome_pred)
cfm
classification_accuracy = sum(diag(cfm))/sum(cfm)
classification_accuracy
