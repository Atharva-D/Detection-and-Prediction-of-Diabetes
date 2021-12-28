library(stats)
library(dplyr)
library(randomForest)
mydata<- read.csv("IRIS.CSV")
View(mydata)
str(mydata)
index <-sample(2,nrow(mydata),replace = TRUE,prob = c(0.7,0.3))
#traing data
set.seed(123)
training <-mydata[index==1,]
testing <-mydata[index==2,]


#random forest model
str(mydata)
#rfm = randomForest(species~ sepal_length+sepal_width+petal_length+petal_width,data=training)

rfm<-randomForest(as.factor(species)~.,training, mtry = 3,
                  importance = TRUE, na.action = na.omit)
#rfm<-randomForest(species~.,training)
species_pred<-predict(rfm,testing)
testing$species_pred=species_pred


cfm<-table(testing$species,testing$species_pred)
cfm
classification_accuracy = sum(diag(cfm))/sum(cfm)
classification_accuracy
varImpPlot(species_pred)

