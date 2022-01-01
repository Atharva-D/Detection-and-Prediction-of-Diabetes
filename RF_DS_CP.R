library(stats)
library(dplyr)
library(randomForest)
library(groupdata2)
library(caret)
library(data.table) 
library(dplyr) 
library(ggplot2) 
library(caret) 
library(xgboost) 
library(e1071) 
library(cowplot) 
library(matrix)
library(pROC) #ROC

library(magrittr)
library(glmnet)


library(groupdata2)

data = read.csv("data.csv")
print(str(data))
data<-upsample(
  data,
  cat_col= "Outcome",
  id_col = NULL,
  id_method = "n_ids",
  mark_new_rows = FALSE
)
data$Outcome <- as.factor(data$Outcome)
table(data$Outcome)
set.seed(222)
ind <- sample(2, nrow(data), replace = TRUE,prob = c(0.8,0.2))
train <- data[ind==1,]
test <- data[ind==2,]
set.seed(123)
rf <- randomForest(Outcome~., data = train,
                   ntree = 800, 
                   mtry = 2,nodesize=5,
                   importance = TRUE,
                   proximity = TRUE)
p1 <- predict(rf,train)
confusionMatrix(p1, train$Outcome)


p2 <- predict(rf,test)
confusionMatrix(p2,test$Outcome)

pred_prob_rf <- predict(rf, test, type="prob")
print(pred_prob_rf )
print()
roc_rf <- roc(test$Outcome, pred_prob_rf$pos)




set.seed(222)
fitControl = trainControl(
  method = 'cv' , 
  number = 10 ,
  savePredictions = 'final' ,
  classProbs = T,
  summaryFunction = twoClassSummary
)

model_forest <- caret::train(Outcome ~., data = train,
                             method = "ranger",
                        
                             metric = "ROC",
                             trControl = trainControl(method = "cv", number = 10,
                                                      classProbs = T))
#preProcess = c("center","scale","pca"))
model_forest
plot(model_forest)
pred_rf <- predict(model_forest, test)
# Confusion Matrix 
cm_rf <- confusionMatrix(pred_rf, test$Outcome, positive="pos")

# Prediction Probabilities
pred_prob_rf <- predict(model_forest, test, type="prob")
# ROC value
roc_rf <- roc(test$Outcome, pred_prob_rf$pos)

# Confusion Matrix for Random Forest Model
cm_rf
print(roc_rf)
caTools::colAUC(pred_prob_rf$pos, test$Outcome, plotROC = T)
