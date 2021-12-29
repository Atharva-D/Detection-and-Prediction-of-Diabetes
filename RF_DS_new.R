#install.packages("randomForest")
#install.packages("caret")
data <- read.csv("diabetes_new.csv")
str(data)

data$Outcome <- as.factor(data$Outcome)
table(data$Outcome)

set.seed(123)
ind <- sample(2, nrow(data), replace = TRUE,prob = c(0.7,0.3))
train <- data[ind==1,]
test <- data[ind==2,]


library(randomForest)
set.seed(222)
rf <- randomForest(Outcome~., data = train,
                   ntree = 550, 
                   mtry = 2, 
                   importance = TRUE, 
                   proximity = TRUE)
print(rf)
attributes(rf)


library(caret)
p1 <- predict(rf,train)
confusionMatrix(p1, train$Outcome)


p2 <- predict(rf,test)
confusionMatrix(p2,test$Outcome)


plot(rf)


t <- tuneRF(train[,-9], train[,9],
            stepFactor = 6,
            plot = TRUE,
            ntreeTry = 800,
            trace = TRUE,
            improve = 0.2)
