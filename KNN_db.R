library(caret) #ML Model buidling package
library(tidyverse) #ggplot and dplyr
library(MASS) #Modern Applied Statistics with S
library(mlbench) #data sets from the UCI repository.
library(summarytools)
library(corrplot) #Correlation plot
library(gridExtra) #Multiple plot in single grip space
library(timeDate) 
library(ggplot2)
library(pROC) #ROC
library(caTools) #AUC
library(rpart.plot) #CART Decision Tree
library(e1071) #imports graphics, grDevices, class, stats, methods, utils
library(graphics)#fourfoldplot
library(groupdata2)
data(PimaIndiansDiabetes)
df <- PimaIndiansDiabetes
df=read.csv("diabetes.csv")
df<-upsample(
  df,
  cat_col= "Outcome",
  id_col = NULL,
  id_method = "n_ids",
  mark_new_rows = FALSE
)


str(df)
head(df)
#store rows for partition
partition <- caret::createDataPartition(y = df$Outcome, times = 1, p = 0.7, list = FALSE)

# create training data set
train_set <- df[partition,]
print(train_set)
# create testing data set, subtracting the rows partition to get remaining 30% of the data
test_set <- df[-partition,]
print(test_set)
str(train_set)
str(test_set)

summary(train_set)
summarytools::descr(train_set)


ggplot(df, aes(df$diabetes, fill = diabetes)) + 
  geom_bar() +
  theme_bw() +
  labs(title = "Diabetes Classification", x = "Diabetes") +
  theme(plot.title = element_text(hjust = 0.5))
table(df$Outcome)
cor_data <- cor(train_set[,setdiff(names(train_set), 'Outcome')])
#Numerical Correlation Matrix
cor_data
corrplot::corrplot(cor_data)
corrplot::corrplot(cor_data, type = "lower", method = "number")

model_knn <- caret::train(diabetes ~., data = train_set,
                          method = "knn",
                          metric = "ROC",
                          tuneGrid = expand.grid(.k = c(3:10)),
                          trControl = trainControl(method = "cv", number = 10,
                                                   classProbs = T, summaryFunction = twoClassSummary),
                          preProcess = c("center","scale","pca"))

model_knn
plot(model_knn)
model_knn$results[7,2]
pred_knn <- predict(model_knn, test_set)
# Confusion Matrix 
cm_knn <- confusionMatrix(pred_knn, test_set$diabetes, positive="pos")

# Prediction Probabilities
pred_prob_knn <- predict(model_knn, test_set, type="prob")
# ROC value
roc_knn <- roc(test_set$diabetes, pred_prob_knn$pos)

# Confusion matrix 
cm_knn
roc_knn
caTools::colAUC(pred_prob_knn$pos, test_set$diabetes, plotROC = T)

fitControl = trainControl(
  method = 'cv' , 
  number = 10 ,
  savePredictions = 'final' ,
  classProbs = T,
  summaryFunction = twoClassSummary
)

model_svm = caret::train(diabetes ~ . , data=train_set, method = 'svmRadial' ,metric="ROC", tuneLength = 5, trControl = fitControl) #SVM MODEL



model_svm
plot(model_svm)
pred_svm <- predict(model_svm, test_set)
# Confusion Matrix 
cm_svm <- confusionMatrix(pred_svm, test_set$diabetes, positive="pos")
pred_prob_svm <- predict(model_svm, test_set, type="prob")
# ROC value
roc_svm <- roc(test_set$diabetes, pred_prob_svm$pos)
cm_svm
roc_svm
caTools::colAUC(pred_prob_svm$pos, test_set$diabetes, plotROC = T)



model_forest <- caret::train(diabetes ~., data = train_set,
                             method = "ranger",
                             metric = "ROC",
                             trControl = trainControl(method = "cv", number = 10,
                                                      classProbs = T, summaryFunction = twoClassSummary))
                             #preProcess = c("center","scale","pca"))
model_forest
plot(model_forest)
pred_rf <- predict(model_forest, test_set)
# Confusion Matrix 
cm_rf <- confusionMatrix(pred_rf, test_set$diabetes, positive="pos")

# Prediction Probabilities
pred_prob_rf <- predict(model_forest, test_set, type="prob")
# ROC value
roc_rf <- roc(test_set$diabetes, pred_prob_rf$pos)

# Confusion Matrix for Random Forest Model
cm_rf

xgb_grid_1  <-  expand.grid(
  nrounds = 20,
  eta = 0.5,
  max_depth = 20,
  gamma = 0,
  colsample_bytree = 0.6,
  min_child_weight = 1,
  subsample = 0.5
)

model_xgb <- caret::train(diabetes ~., data = train_set,
                          method = "xgbTree",
                          metric = "ROC",
                          tuneGrid=xgb_grid_1,
                          trControl = trainControl(method = "cv", number = 10,
                                                   classProbs = T, summaryFunction = twoClassSummary),
                          preProcess = c("center","scale","pca"))
model_xgb
model_xgb$results["ROC"]
#plot(model_xgb)


pred_xgb <- predict(model_xgb, test_set)
# Confusion Matrix 
cm_xgb <- confusionMatrix(pred_xgb, test_set$diabetes, positive="pos")

# Prediction Probabilities
pred_prob_xgb <- predict(model_xgb, test_set, type="prob")
# ROC value
roc_xgb <- roc(test_set$diabetes, pred_prob_xgb$pos)

# Confusion matrix 
cm_xgb
roc_xgb
caTools::colAUC(pred_prob_xgb$pos, test_set$diabetes, plotROC = T)




sparse_matrix <- sparse.model.matrix(Outcome ~ ., data = train_set)[,-1]
library(data.table) 
library(dplyr) 
library(ggplot2) 
library(caret) 
library(xgboost) 
library(e1071) 
library(cowplot) 
library(matrix)


library(magrittr)
library(glmnet)


library(groupdata2)

y_train<-train_set$Outcome
bst <- xgboost(data = sparse_matrix, label = y_train, max_depth = 20, eta = 0.5, nthread = -1,nrounds = 20, objective = "binary:logistic")
y_test<-test_set$Outcome
test_n <- sparse.model.matrix(Outcome ~ ., data = test_set)[,-1]
pred <- predict(bst, test_n)
# get & print the classification error
err <- mean(as.numeric(pred > 0.5) != y_test)
print(paste("test-error=", err))
accuracy = 1-err
print(paste("Accuracy of XGBoost model is:",accuracy))
