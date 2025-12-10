# Analysis of Breast Cancer Wisconsin (Diagnostic) Data from "The UCI Machine Learning Repository"
## Capstone Project for "Choose your own project" 

## Load the needed packages
library(tidyverse)
library(caret)
library(randomForest)
library(ggplot2)
library(GGally)
library(pROC)
library(reshape2)
library(ggcorrplot)
library(gt)
set.seed(123)

## Load the data
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
bc_data <- read.csv(url, header = FALSE)

##check the data structure

head(bc_data)

#### ***************************************************************************
# Variable Information from the Dataset info at UCI site
# 
# 1) ID number
# 2) Diagnosis (M = malignant, B = benign)
# 3-32)
# 
# Ten real-valued features are computed for each cell nucleus:
#   
#   a) radius (mean of distances from center to points on the perimeter)
# b) texture (standard deviation of gray-scale values)
# c) perimeter
# d) area
# e) smoothness (local variation in radius lengths)
# f) compactness (perimeter^2 / area - 1.0)
# g) concavity (severity of concave portions of the contour)
# h) concave points (number of concave portions of the contour)
# i) symmetry 
# j) fractal dimension ("coastline approximation" - 1)

## *****************************************************************************

## Use the above info from the dataset to name the variables for better understanding

names(bc_data) <- c(
  "id", "diagnosis",
  paste0(rep(c("radius", "texture", "perimeter", "area", "smoothness",
               "compactness", "concavity", "concave_points", 
               "symmetry", "fractal_dimension"), each = 3),
         rep(c("_mean", "_se", "_worst"), 10))
)

str(bc_data)
table(bc_data$diagnosis)
##---------------
# Prepare the data for Analysis
# Convert diagnosis to factor
bc_data$diagnosis <- factor(bc_data$diagnosis, levels = c("B", "M"))
# Remove 'id' column since we do not need that
bc_data <- bc_data %>% select(-id)

## Exploratory Analysis of data

ggplot(bc_data, aes(diagnosis)) +
  geom_bar(fill = c("cyan", "red")) +
  ggtitle("Benign vs Malignant Tumor Distribution")

summary(bc_data)

## correlation heatmap
bc_data_mean <- bc_data %>% select(contains("mean"), diagnosis)
corr <- cor(bc_data_mean %>% select(-diagnosis))
ggcorrplot::ggcorrplot(corr)


##Principal Component Analysis (PCA)

pca <- prcomp(bc_data %>% select(-diagnosis), scale. = TRUE)
pca_df <- data.frame(pca$x[,1:2], diagnosis = bc_data$diagnosis)

ggplot(pca_df, aes(PC1, PC2, color = diagnosis)) +
  geom_point(alpha = .7) +
  ggtitle("PCA: PC1 vs PC2")

## split the data into train and test
index <- createDataPartition(bc_data$diagnosis, p = 0.8, list = FALSE)
train <- bc_data[index, ]
test <- bc_data[-index, ]

## Model 1 - Logistic regression
log_fit <- train(
  diagnosis ~ ., 
  data = train,
  method = "glm"
)

# Classification predictions
log_pred <- predict(log_fit, test)
log_cm <- confusionMatrix(log_pred, test$diagnosis)

# Probabilities for ROC/AUC
log_prob <- predict(log_fit, test, type = "prob")[, "M"]

# ROC curve and AUC
roc_obj <- roc(test$diagnosis, log_prob)
plot(roc_obj, col = "red",lwd = 3, main = "ROC Curve - Logistic Regression")
log_auc <- auc(roc_obj)

log_cm
log_auc


##Model 2 K- Nearest Neighbors

# Train KNN
knn_fit <- train(
  diagnosis ~ ., data = train,
  method = "knn",
  tuneGrid = data.frame(k = seq(3, 21, 2)),
  trControl = trainControl(method = "cv", number = 10),
  preProcess = c("center", "scale")   # VERY IMPORTANT FOR KNN
)

# Predictions (class)
knn_pred <- predict(knn_fit, test)
knn_cm <- confusionMatrix(knn_pred, test$diagnosis)

# Probabilities for ROC
knn_prob <- predict(knn_fit, test, type = "prob")[, "M"]

roc_obj <- roc(test$diagnosis, knn_prob)
plot(roc_obj, col = "blue",lwd = 3, main = "ROC Curve - KNN")
knn_auc <- auc(roc_obj)

knn_cm
knn_auc
##Model 3 Random Forest

rf_fit <- train(
  diagnosis ~ ., data = train,
  method = "rf",
  trControl = trainControl(method = "cv", number = 10),
  importance = TRUE
)

# Train Random Forest
rf_fit <- train(
  diagnosis ~ ., 
  data = train,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5),
  importance = TRUE
)

# Predictions (class labels)
rf_pred <- predict(rf_fit, test)
rf_cm <- confusionMatrix(rf_pred, test$diagnosis)

# Probabilities for ROC
rf_prob <- predict(rf_fit, test, type = "prob")[, "M"]

# ROC Curve and AUC
roc_obj <- roc(test$diagnosis, rf_prob)
plot(roc_obj, col = "darkgreen", lwd = 3, main = "ROC Curve - Random Forest")
rf_auc <- auc(roc_obj)

rf_cm
rf_auc


## Model 4 - SVM

# Train SVM (Radial Kernel)
svm_fit <- train(
  diagnosis ~ ., 
  data = train,
  method = "svmRadial",
  trControl = trainControl(method = "cv", number = 10,classProbs = TRUE),
  tuneLength = 10
)

# Predictions
svm_pred <- predict(svm_fit, test)
svm_cm <- confusionMatrix(svm_pred, test$diagnosis)

# Probabilities + ROC
svm_prob <- predict(svm_fit, test, type = "prob")[,"M"]
roc_obj <- roc(test$diagnosis, svm_prob)

plot(roc_obj, col = "purple", lwd = 3, main = "ROC Curve - SVM")
svm_auc <- auc(roc_obj)

svm_cm
svm_auc

 
### Model comparison
results <- data.frame(
  Model = c("Logistic Regression","KNN","Random Forest","SVM"),
  Accuracy = c(log_cm$overall["Accuracy"],
               knn_cm$overall["Accuracy"],
               rf_cm$overall["Accuracy"],
               svm_cm$overall["Accuracy"]),
  Sensitivity = c(log_cm$byClass["Sensitivity"],
                  knn_cm$byClass["Sensitivity"],
                  rf_cm$byClass["Sensitivity"],
                  svm_cm$byClass["Sensitivity"]),
  Specificity = c(log_cm$byClass["Specificity"],
                  knn_cm$byClass["Specificity"],
                  rf_cm$byClass["Specificity"],
                  svm_cm$byClass["Specificity"]),
  AUC = c(log_auc, knn_auc, rf_auc, svm_auc)
)

## gt table
results %>%
  gt() %>%
  tab_header(
    title = "Model Performance Comparison",
    subtitle = "Accuracy, Sensitivity, Specificity, and AUC "
  ) %>%
  fmt_number(
    columns = c(Accuracy, Sensitivity, Specificity, AUC),
    decimals = 3
  )


