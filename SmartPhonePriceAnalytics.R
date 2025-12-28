# working directory
#getwd()
#setwd("/Users/sdahal/GitRepos/ML_Used_Smartphone_Price_Analytics")

#install.packages("skimr")
library(ggplot2)    
library(dplyr)      
library(tidyr)     
library(corrplot)   
library(e1071)
library(randomForest)
library(psych)
library(VIM) # missing value visualization
library(psych) 
library(car)
library(caret)
library(splitTools) # for splitting datasets
library(FNN)   
library(randomForest)
library(glmnet) 

# reproducibility
set.seed(123)

# Loading
used_device_data <- read.csv("dataset/used_device_data.csv")
str(used_device_data)

# Preview
head(used_device_data)

# summary statistics for numeric variables
numeric_vars <- used_device_data %>% select(where(is.numeric))
summary(numeric_vars)

# histograms for numeric variables
# Set up plotting area
par(mfrow = c(2, 3))
hist(numeric_vars$screen_size, main = "Screen Size Distribution", xlab = "Screen Size (cm)", col = "lightblue")
hist(numeric_vars$internal_memory, main = "Internal Memory Distribution", xlab = "Internal Memory (GB)", col = "lightblue")
hist(numeric_vars$ram, main = "RAM Distribution", xlab = "RAM (GB)", col = "lightblue")
hist(numeric_vars$battery, main = "Battery Distribution", xlab = "Battery (mAh)", col = "lightblue")
hist(numeric_vars$days_used, main = "Days Used Distribution", xlab = "Days Used", col = "lightblue")
hist(numeric_vars$normalized_used_price, main = "Normalized Used Price Distribution", xlab = "Normalized Used Price", col = "lightblue")

# categorical data--------------------
# frequency tables with percentages
round(prop.table(table(used_device_data$device_brand)) * 100, 2)
round(prop.table(table(used_device_data$os)) * 100, 2)
round(prop.table(table(used_device_data$X4g)) * 100, 2)
round(prop.table(table(used_device_data$X5g)) * 100, 2)

# categorical data visualization--------------------
ggplot(used_device_data, aes(x = device_brand)) +
  geom_bar(fill = "steelblue") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Device Brand Distribution", x = "Brand", y = "Count")

# bar plot for os
ggplot(used_device_data, aes(x = os)) +
  geom_bar(fill = "skyblue") +
  theme_minimal() +
  labs(title = "Operating System Distribution", x = "OS", y = "Count")

# relation ship between new and old normalized price 
pairs.panels(used_device_data[, c("normalized_new_price", "normalized_used_price")])

##Data Preprocessing###############################
################### checking missing value
missing_summary <- colSums(is.na(used_device_data))
data.frame(Missing_Count = missing_summary, 
           Missing_Percent = paste0(round((missing_summary / nrow(used_device_data)) * 100, 2), "%"))

## combintation of missing 
short_labels <- abbreviate(names(used_device_data), minlength = 10)
aggr(used_device_data, 
     col = c("skyblue", "red"), 
     numbers = TRUE, 
     prop = FALSE, 
     sortVars = TRUE, 
     labels = short_labels, 
     cex.axis = 0.7,
     gap = 3, 
     ylab = c("Number of Missing", "Combination"))

# find a record of missing battery and ram

missing_battery_ram_record <- used_device_data[is.na(used_device_data$battery) & is.na(used_device_data$ram), ] 
missing_battery_ram_record

# total missing of missing battery and ram
missing_battery_ram_before <- sum(is.na(used_device_data$battery) & is.na(used_device_data$ram))
cat("Number of rows with both battery and RAM missing:", missing_battery_ram_before)

# omit record of missing battery and ram
used_device_data <- used_device_data[!(is.na(used_device_data$battery) & is.na(used_device_data$ram)), ]
missing_battery_ram_after <- sum(is.na(used_device_data$battery) & is.na(used_device_data$ram))
cat("verifying after removal:", missing_battery_ram_after)

dim(used_device_data)

### missing handling ---- impute
# List of numeric columns to impute
numeric_cols <- c("rear_camera_mp", "front_camera_mp", "internal_memory", 
                  "ram", "battery", "weight")

# Loop through each column and impute missing values
for (col in numeric_cols) {
  
  # Brand wise median imputation
  used_device_data <- used_device_data %>%
    group_by(device_brand) %>%
    mutate(!!sym(col) := ifelse(is.na(.data[[col]]),
                                median(.data[[col]], na.rm = TRUE),
                                .data[[col]])) %>%
    ungroup()
  
  # Fill any remaining NAs with overall median
  used_device_data[[col]][is.na(used_device_data[[col]])] <- 
    median(used_device_data[[col]], na.rm = TRUE)
}

# verifying no missing values remain
missing_summary_verify <- colSums(is.na(used_device_data))
data.frame(Missing_Count = missing_summary_verify, 
           Missing_Percent = paste0(round((missing_summary_verify / nrow(used_device_data)) * 100, 2), "%"))

# Check for empty strings
sum(used_device_data == "", na.rm = TRUE)

# checking zeros entire data set except categorical variables 
zeros_summary <- colSums(used_device_data[sapply(used_device_data, is.numeric)] == 0, na.rm = TRUE)
data.frame(Zero_Count = zeros_summary,
           Zero_Percent = paste0(round((zeros_summary / nrow(used_device_data)) * 100, 2), "%"))

##### 
# let me check by which brand and year having front_camera_mp = 0
used_device_data %>%
  filter(front_camera_mp == 0) %>%
  group_by(device_brand, release_year) %>%
  summarise(n = n(), .groups = "drop")

# front_camera_mp in the Nokia phone are  valid

######################## Outliers Detection
numeric_cols <- used_device_data[sapply(used_device_data, is.numeric)]

# Function to calculate outlier count using IQR
find_outliers <- function(x) {
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  sum(x < (q1 - 1.5*iqr) | x > (q3 + 1.5*iqr), na.rm = TRUE)
}

# apply function to all numeric columns
outlier_counts <- sapply(numeric_cols, find_outliers)

# create summary table without duplicating column names
outliers_df <- data.frame(
  Variable = names(outlier_counts),
  Outlier_Count = as.numeric(outlier_counts),
  Min_Value = round(sapply(numeric_cols, function(x) min(x, na.rm = TRUE)), 2),
  Max_Value = round(sapply(numeric_cols, function(x) max(x, na.rm = TRUE)), 2),
  Percentage = paste0(round(as.numeric(outlier_counts) / nrow(used_device_data) * 100, 2), "%"),
  row.names = NULL
)

print(outliers_df)

######################## Outliers Handling
high_impact_cols <- c("screen_size", "weight")

handle_outliers <- function(x) {
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower_bound <- q1 - 1.5 * iqr
  upper_bound <- q3 + 1.5 * iqr
  #replaceing outliers with nearest bound
  x[x < lower_bound] <- lower_bound
  x[x > upper_bound] <- upper_bound
  return(x)
}

used_device_data[high_impact_cols] <- lapply(used_device_data[high_impact_cols], handle_outliers)

# Verify results after outlier handling
outlier_counts <- sapply(used_device_data[high_impact_cols], find_outliers)

# Summary table
outlier_handle_df <- data.frame(
  Variable = names(outlier_counts),
  Outlier_Count = as.numeric(outlier_counts),
  Min_Value = round(sapply(high_impact_cols, function(x) min(used_device_data[[x]], na.rm = TRUE)), 2),
  Max_Value = round(sapply(high_impact_cols, function(x) max(used_device_data[[x]], na.rm = TRUE)), 2),
  Percentage = paste0(round(as.numeric(outlier_counts) / nrow(used_device_data) * 100, 2), "%"),
  row.names = NULL
)
# print result
print(outlier_handle_df)

# create Price Categories
#Business rule based on features

used_device_data$price_category <- ifelse(
  used_device_data$ram >= 4 &
    used_device_data$rear_camera_mp >= 8 &
    used_device_data$internal_memory >= 64 &
    used_device_data$release_year >= 2017,
  "High", 
  "Low"
)
#table(used_device_data$price_category)

# Predictor Analysis and Relevancy
#View(used_device_data)

# Correlation Analysis for Numeric Predictors
library(dplyr)
numeric_data <- used_device_data %>%
  select(screen_size, rear_camera_mp, front_camera_mp,
         internal_memory, ram, battery, weight,
         release_year, days_used, normalized_new_price,
         normalized_used_price)

# Calculate correlation matrix
cor_matrix <- cor(numeric_data, use = "complete.obs")
cor_matrix

options(repr.plot.width = 16, repr.plot.height = 14)  
par(mfrow = c(1, 1), mar = c(2, 2, 2, 2))

# Increase text size
corrplot(cor_matrix,
         method = "color",
         type = "upper",
         tl.cex = 1.0,        # larger variable names
         tl.col = "black",
         addCoef.col = "black",
         number.cex = 1.2,    # larger correlation numbers
         col = colorRampPalette(c("blue", "white", "red"))(200))

# Pairs Plot of Numeric Variables
pairs.panels(
  used_device_data[, sapply(used_device_data, is.numeric)],
  #cex.labels = 1.0,   # Increase size of labels inside the plot
  font.labels = 2,    # Make labels bold (optional)
  cex = 1.1         # Increase axis text size
)

# vif Function 
#categorical variables are factors
used_device_data_vif <- used_device_data
used_device_data_vif <- used_device_data_vif %>% select(-all_of("price_category"))
used_device_data_vif$device_brand <- as.factor(used_device_data_vif$device_brand)
used_device_data_vif$os <- as.factor(used_device_data_vif$os)
used_device_data_vif$X4g <- as.factor(used_device_data_vif$X4g)
used_device_data_vif$X5g <- as.factor(used_device_data_vif$X5g)

#detecting multicollinearity
# car::vif() function to assess multicollinearity
model_vif <- lm(normalized_used_price ~ ., data=used_device_data_vif)
vif_values <- vif(model_vif)
print(vif_values)

################################ Categorical predictor relevancy
used_device_data_cpr <- used_device_data
used_device_data_cpr$device_brand <- as.factor(used_device_data_cpr$device_brand)
used_device_data_cpr$os <- as.factor(used_device_data_cpr$os)

# Chi-Square Test
table_brand_os <- table(used_device_data_cpr$device_brand, used_device_data_cpr$os)
chisq_test <- chisq.test(table_brand_os)
print(chisq_test)

# Dimension reduction (if needed)
#now pca
numeric_predictors <- used_device_data %>%
  select(screen_size, rear_camera_mp, front_camera_mp, internal_memory, 
         ram, battery, weight, release_year, days_used, normalized_new_price)
numeric_predictors_scaled <- scale(numeric_predictors)
pca_result <- prcomp(numeric_predictors_scaled, center = TRUE, scale. = TRUE)
summary(pca_result)

# ---identified first 4 PCA components (PC1-PC4) since they explain ~80.65% of the variance.
#  analyzed PCA summary result but not decided to use

# Data Transformation
# #############Data Partition 
strata_factor <- interaction(
  used_device_data$device_brand,
  used_device_data$os,
  used_device_data$X5g,
  used_device_data$price_category,
  drop = TRUE
)
# perform a stratified 60/20/20 split
set.seed(2025)
splits <- partition(
  y = strata_factor, 
  p = c(train = 0.6, val = 0.2, test = 0.2),
  type = "stratified"
)
# Extract partitions
train_data <- used_device_data[splits$train, ]
val_data   <- used_device_data[splits$val, ]
test_data  <- used_device_data[splits$test, ]

###################### let create summary
# Partition sizes
train_n <- nrow(train_data)
val_n   <- nrow(val_data)
test_n  <- nrow(test_data)

# Total records
total_n <- nrow(used_device_data)

# Create summary table
partition_summary <- data.frame(
  Partition   = c("Training", "Validation", "Test"),
  Records     = c(train_n, val_n, test_n),
  Percentage  = round(c(train_n, val_n, test_n) / total_n * 100, 2)
)
print(partition_summary)

#Verify distributions
round(prop.table(table(train_data$os)) * 100, 2)
round(prop.table(table(val_data$os)) * 100, 2)
round(prop.table(table(test_data$os)) * 100, 2)

round(prop.table(table(train_data$X5g)) * 100, 2)
round(prop.table(table(val_data$X5g)) * 100, 2)
round(prop.table(table(test_data$X5g)) * 100, 2)

# Check overlap between splits
length(intersect(splits$train, splits$test))
length(intersect(splits$train, splits$val)) 
length(intersect(splits$val, splits$test))

# Check coverage
length(unique(c(splits$train, splits$val, splits$test))) == nrow(used_device_data)

# Selected predictors based on correlation & relevance
#  including normalized_new_price for observation
#features <- c("screen_size", "rear_camera_mp", "front_camera_mp", "ram", "battery", "days_used", "release_year","device_brand", "X4g", "X5g", "normalized_new_price")
features <- c("screen_size", "rear_camera_mp", "front_camera_mp", "ram", "battery", "days_used", "release_year","device_brand", "X4g", "X5g")

#  predictors & target variable for regression
X_train <- train_data %>% select(all_of(features))
y_train <- train_data$normalized_used_price

X_val <- val_data %>% select(all_of(features))
y_val <- val_data$normalized_used_price

X_test <- test_data %>% select(all_of(features))
y_test <- test_data$normalized_used_price

# predictors & target variable for classification
X_train_cat <- train_data %>% select(all_of(features))
y_train_cat <- train_data$price_category

X_val_cat <- val_data %>% select(all_of(features))
y_val_cat <- val_data$price_category

X_test_cat <- test_data %>% select(all_of(features))
y_test_cat <- test_data$price_category

# convert to factor for categorical target
y_train_cat <- factor(ifelse(y_train_cat == "High", 1, 0), levels = c(0, 1))
y_val_cat   <- factor(ifelse(y_val_cat == "High", 1, 0), levels = c(0, 1))
y_test_cat  <- factor(ifelse(y_test_cat == "High", 1, 0), levels = c(0, 1))
# Convert 4G/5G to Binary

binary_conversion <- function(df) {
  df$X4g <- as.numeric(df$X4g == "yes")
  df$X5g <- as.numeric(df$X5g == "yes")
  return(df)
}
# binary conversion for regression
X_train <- binary_conversion(X_train)
X_val   <- binary_conversion(X_val)
X_test  <- binary_conversion(X_test)

# binary conversion for classification
X_train_cat <- binary_conversion(X_train_cat)
X_val_cat   <- binary_conversion(X_val_cat)
X_test_cat  <- binary_conversion(X_test_cat)

# Evaluation Function for all model
calculate_metrics <- function(actual, predicted) {
  rmse <- sqrt(mean((actual - predicted)^2))
  mae  <- mean(abs(actual - predicted))
  r2   <- 1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
  return(c(RMSE = rmse, MAE = mae, R2 = r2))
}

# Define features
numerical_features_knn <- c("screen_size", "rear_camera_mp", "front_camera_mp", "ram", "battery", "days_used", "release_year")
binary_features <- c("X4g", "X5g")
#categorical_features <- c("device_brand", "os")
categorical_features <- c("device_brand")

##################################### kNN Regression
X_train_knn <- X_train
X_val_knn   <- X_val
X_test_knn  <- X_test

#str(X_train_knn)

#View(X_train_knn)
#One-hot encode device_brand (only categorical variable) ===
dummy <- dummyVars(~ device_brand, data = X_train_knn)
brand_train <- predict(dummy, newdata = X_train_knn) %>% as.data.frame()
brand_val   <- predict(dummy, newdata = X_val_knn)   %>% as.data.frame()
brand_test  <- predict(dummy, newdata = X_test_knn)  %>% as.data.frame()


# Remove original device_brand column and bind one-hot columns
X_train_knn <- X_train_knn %>% select(-device_brand) %>% cbind(brand_train)
X_val_knn <- X_val_knn %>% select(-device_brand) %>% cbind(brand_val)
X_test_knn <- X_test_knn %>% select(-device_brand) %>% cbind(brand_test)


# === Z-SCORE STANDARDIZATION 
scaler <- preProcess(X_train_knn, method = c("center", "scale"))
X_train_knn <- predict(scaler, X_train_knn) %>% as.matrix()
X_val_knn   <- predict(scaler, X_val_knn)   %>% as.matrix()
X_test_knn  <- predict(scaler, X_test_knn)  %>% as.matrix()

# initial k
initial_k <- 3
# Validation predictions
set.seed(2025)
knn_val_init <- knn.reg(train = X_train_knn,
                        test  = X_val_knn,
                        y     = y_train,
                        k     = initial_k)
y_val_pred_init <- knn_val_init$pred
val_metrics_init <- calculate_metrics(y_val, y_val_pred_init)
cat("validation metrics initial k =", initial_k)
print(val_metrics_init)

# Test predictions
set.seed(2025)
knn_test_init <- knn.reg(train = X_train_knn,
                         test  = X_test_knn,
                         y     = y_train,
                         k     = initial_k)

y_test_pred_init <- knn_test_init$pred
test_metrics_init <- calculate_metrics(y_test, y_test_pred_init)
cat("test metrics initial k =", initial_k)
print(test_metrics_init)

##################################### Tune kNN Regression
set.seed(2025)
k_values <- seq(1, 30, by = 2)  
val_rmse <- numeric(length(k_values))

for (i in seq_along(k_values)) {
  k <- k_values[i]
  knn_pred_val <- knn.reg(train = X_train_knn, test = X_val_knn, y = y_train, k = k)$pred
  val_rmse[i] <- sqrt(mean((y_val - knn_pred_val)^2))
  cat("k = ", k, " Validation RMSE = ", round(val_rmse[i], 4), "\n")
}
best_k_idx <- which.min(val_rmse)
best_k <- k_values[best_k_idx]
best_k

#predictions with Best k

# Validation set
knn_val_best <- knn.reg(train = X_train_knn,
                        test  = X_val_knn,
                        y     = y_train,
                        k     = best_k)

y_val_pred_best <- knn_val_best$pred
val_metrics_best <- calculate_metrics(y_val, y_val_pred_best)
cat("Validation Metrics (Best k =", best_k, "):\n")
print(val_metrics_best)

# Test set
knn_test_best <- knn.reg(train = X_train_knn,
                         test  = X_test_knn,
                         y     = y_train,
                         k     = best_k)

y_test_pred_best <- knn_test_best$pred
test_metrics_best <- calculate_metrics(y_test, y_test_pred_best)
cat("Test Metrics (Best k =", best_k, "):\n")
print(test_metrics_best)

##################################### k-NN Classifier
X_train_knn_cls <- X_train_cat
X_val_knn_cls <- X_val_cat
X_test_knn_cls <- X_test_cat

#One-hot encode device_brand (only categorical variable) ===
dummy_cls <- dummyVars(~ device_brand, data = X_train_knn_cls)
brand_train_cls <- predict(dummy_cls, newdata = X_train_knn_cls) %>% as.data.frame()
brand_val_cls   <- predict(dummy_cls, newdata = X_val_knn_cls)   %>% as.data.frame()
brand_test_cls  <- predict(dummy_cls, newdata = X_test_knn_cls)  %>% as.data.frame()

# Remove original device_brand column and bind one-hot columns
X_train_knn_cls <- X_train_knn_cls %>% select(-device_brand) %>% cbind(brand_train_cls)
X_val_knn_cls <- X_val_knn_cls %>% select(-device_brand) %>% cbind(brand_val_cls)
X_test_knn_cls <- X_test_knn_cls %>% select(-device_brand) %>% cbind(brand_test_cls)

# === Z-SCORE STANDARDIZATION 
scaler_cls <- preProcess(X_train_knn_cls, method = c("center", "scale"))
X_train_knn_cls <- predict(scaler_cls, X_train_knn_cls) %>% as.matrix()
X_val_knn_cls   <- predict(scaler_cls, X_val_knn_cls)   %>% as.matrix()
X_test_knn_cls  <- predict(scaler_cls, X_test_knn_cls)  %>% as.matrix()

#View(X_train_knn_cls)
# initial kNN Performance
# k-NN Classifier using class::knn
initial_cls_k <- 5
set.seed(123)
knn_val_pred <- knn(
  train = X_train_knn_cls,
  test = X_val_knn_cls,
  cl = y_train_cat,
  k = initial_cls_k
)
#align prediction levels with actual
knn_val_pred <- factor(knn_val_pred, levels = levels(y_val_cat))
knn_val_metrics <- confusionMatrix(knn_val_pred, y_val_cat, positive = "1")
cat("Validation Metrics For Classification Initial k =", initial_cls_k)
print(knn_val_metrics)

set.seed(123)
knn_test_pred <- knn(
  train = X_train_knn_cls,
  test = X_test_knn_cls,
  cl = y_train_cat,
  k = initial_cls_k
)
#align prediction levels with actual
knn_test_pred <- factor(knn_test_pred, levels = levels(y_test_cat))
knn_test_metrics <- confusionMatrix(knn_test_pred, y_test_cat, positive = "1")
cat("Test Metrics For Classification (Initial k =", initial_cls_k, "):\n")
print(knn_test_metrics)

############## k-NN Tuning with e1071::tune.knn
set.seed(123)
knntuning_cls <- tune.knn(
  x = X_train_knn_cls,
  y = y_train_cat,
  k = 1:30,
  validation.x = X_val_knn_cls,
  validation.y = y_val_cat
)
print(knntuning_cls)
summary(knntuning_cls)
plot(knntuning_cls, main = "k-NN Tuning: Error vs. k", xlab = "k", ylab = "Error Rate")

# k-NN Classifier with best k
best_k_cls <- knntuning_cls$best.parameters$k
cat("Best k value:", best_k_cls, "\n")

# validation set with best k
set.seed(123)
knn_val_pred_best_cls <- knn(
  train = X_train_knn_cls,
  test = X_val_knn_cls,
  cl = y_train_cat,
  k = best_k_cls
)
#align prediction levels with actual
knn_val_pred_best_cls <- factor(knn_val_pred_best_cls, levels = levels(y_val_cat))
knn_val_metrics_cls <- confusionMatrix(knn_val_pred_best_cls, y_val_cat, positive = "1")
cat("Validation Metrics For Classification (Best k =", best_k_cls, "):\n")
print(knn_val_metrics_cls)

# Test set with best k

set.seed(123)
knn_test_pred_best_cls <- knn(
  train = X_train_knn_cls,
  test = X_test_knn_cls,
  cl = y_train_cat,
  k = best_k_cls
)
#align prediction levels with actual
knn_test_pred_best_cls <- factor(knn_test_pred_best_cls, levels = levels(y_test_cat))
knn_test_metrics_cls <- confusionMatrix(knn_test_pred_best_cls, y_test_cat, positive = "1")
cat("Test Metrics For Classification Best k =", best_k_cls)
print(knn_test_metrics_cls)

##################################### Random Forest Regression
X_train_rf <- X_train
X_val_rf   <- X_val
X_test_rf  <- X_test

# Convert categorical variables to factors
#X_train_rf$device_brand <- as.factor(X_train_rf$device_brand)
for (feature in categorical_features) {
  levels_all <- unique(c(X_train_rf[[feature]], X_val_rf[[feature]], X_test_rf[[feature]]))
  X_train_rf[[feature]] <- factor(X_train_rf[[feature]], levels = levels_all)
  X_val_rf[[feature]]   <- factor(X_val_rf[[feature]], levels = levels_all)
  X_test_rf[[feature]]  <- factor(X_test_rf[[feature]], levels = levels_all)
}
#str(X_train_rf$device_brand)
#View(X_train_rf)
# Model Training & Evaluation
#Random Forest
set.seed(123)
rf_model <- randomForest(y_train ~ ., 
                         data = data.frame(X_train_rf, y_train), 
                         ntree = 100,
                         mtry = 4,
                         nodesize = 3
)
# Predict & Evaluate
rf_pred_val  <- predict(rf_model, X_val_rf)
rf_pred_test <- predict(rf_model, X_test_rf)

# Random Forest Metrics
rf_metrics_val  <- calculate_metrics(y_val, rf_pred_val)
rf_metrics_test <- calculate_metrics(y_test, rf_pred_test)

rf_metrics_val
rf_metrics_test

##################################### Random Forest Regression Tuning (Hyperparameter Tuning)
# Hyperparameter grid
p <- ncol(X_train_rf)
rf_grid_rf_Reg <- expand.grid(
  mtry = c(floor(sqrt(p)), floor(p/3), floor(p/2)),
  nodesize = c(3, 5, 7),
  ntree = c(300, 400, 500)
)

results <- data.frame()
for(i in 1:nrow(rf_grid_rf_Reg)) {
  params <- rf_grid_rf_Reg[i, ]
  
  set.seed(123)
  rf <- randomForest(
    x = X_train_rf,
    y = y_train,
    xtest = X_val_rf,
    ytest = y_val,
    mtry     = params$mtry,
    ntree    = params$ntree,
    nodesize = params$nodesize,
    importance = TRUE,
    keep.forest = FALSE
  )
  valid_rmse <- sqrt(rf$test$mse[params$ntree])
  
  results <- rbind(results, data.frame(
    mtry     = params$mtry,
    nodesize = params$nodesize,
    ntree    = params$ntree,
    valid_rmse = valid_rmse
  ))
  cat(sprintf("Row %3d | mtry=%2d nodesize=%2d ntree=%4d RMSE=%.4f\n",
              i, params$mtry, params$nodesize, params$ntree, valid_rmse))
}

# Best model on validation set
best <- results[which.min(results$valid_rmse), ]
best
# Extract best hyperparameters
best_mtry     <- best$mtry
best_nodesize <- best$nodesize
best_ntree    <- best$ntree

# Train final Random Forest model on the training set using best hyperparameters
set.seed(123)
best_model_Reg <- randomForest(
  x = X_train_rf,
  y = y_train,
  mtry     = best_mtry,
  ntree    = best_ntree,
  nodesize = best_nodesize,
  importance = TRUE
)

# train Final Random Forest Model with Best Hyperparameters
# Predict & Evaluate
rf_pred_val_final  <- predict(best_model_Reg, X_val_rf)
rf_metrics_val_final  <- calculate_metrics(y_val, rf_pred_val_final)
print("Tuned Random Forest - Validation Metrics:")
rf_metrics_val_final

rf_pred_test_final <- predict(best_model_Reg, X_test_rf)
rf_metrics_test_final <- calculate_metrics(y_test, rf_pred_test_final)
print("Tuned Random Forest - Test Metrics:")
rf_metrics_test_final

# features 
importance_values <- importance(best_model_Reg) 
importance_df <- data.frame(
  Feature = rownames(importance_values),
  Importance = importance_values[, "IncNodePurity"] 
)
importance_df <- importance_df[order(-importance_df$Importance), ]
print(importance_df)

#####################################  Random Forest Classification
X_train_rf_cls <- X_train_cat
X_val_rf_cls   <- X_val_cat
X_test_rf_cls  <- X_test_cat

# Convert categorical variables to factors
#X_train_rf$device_brand <- as.factor(X_train_rf$device_brand)
for (feature in categorical_features) {
  levels_all <- unique(c(X_train_rf_cls[[feature]], X_val_rf_cls[[feature]], X_test_rf_cls[[feature]]))
  X_train_rf_cls[[feature]] <- factor(X_train_rf_cls[[feature]], levels = levels_all)
  X_val_rf_cls[[feature]]   <- factor(X_val_rf_cls[[feature]], levels = levels_all)
  X_test_rf_cls[[feature]]  <- factor(X_test_rf_cls[[feature]], levels = levels_all)
}
# Random Forest classification
set.seed(123)
rf_model_cls <- randomForest(x=X_train_rf_cls,
                             y=y_train_cat, 
                             ntree = 500,
                             mtry = 4,
                             nodesize = 3,
                             importance = TRUE)
print(rf_model_cls)

# eval random forest on validation set
rf_cls_val_pred <- predict(rf_model_cls, X_val_rf_cls)
rf_cls_val_metrics <- confusionMatrix(rf_cls_val_pred, y_val_cat, positive = "1")
print(rf_cls_val_metrics)

# eval random forest on test set
rf_cls_test_pred <- predict(rf_model_cls, X_test_rf_cls)
rf_cls_test_metrics <- confusionMatrix(rf_cls_test_pred, y_test_cat, positive = "1")
print(rf_cls_test_metrics)

#####################################  Tuned Random Forest Classification
# combine features and target for caret
train_data_rf <- cbind(X_train_rf_cls, price_category = y_train_cat)

# Number of predictors
p1 <- ncol(train_data_rf) - 1 

# hyperparameter grid
tuneGridCs <- expand.grid(
  mtry = c(floor(sqrt(p1)), floor(p1/3), floor(p1/2)),
  splitrule = c("gini"),
  min.node.size = c(3, 5)
)
# Train control
train_control <- trainControl(
  method = "cv",
  number = 5
)
# Number of trees
ntree_values <- c(300, 400, 500)
# Loop over ntree values
best_model <- NULL
best_sensitivity <- -Inf
best_params <- list()
set.seed(123)

for (ntree in ntree_values) {
  cat("Training with ntree =", ntree, "\n")
  rf_model <- train(
    price_category ~ ., 
    data = train_data_rf,
    method = "ranger",
    trControl = train_control,
    tuneGrid = tuneGridCs,
    num.trees = ntree,
    metric = "Sensitivity",
    importance = "impurity"
  )
  
  # Evaluate on validation set
  preds_val <- predict(rf_model, X_val_rf_cls)
  cm_val <- confusionMatrix(preds_val, y_val_cat, positive = "1")
  sensitivity_val <- cm_val$byClass["Sensitivity"]
  
  # Keep best model based on Sensitivity
  if (sensitivity_val > best_sensitivity) {
    best_sensitivity <- sensitivity_val
    best_model <- rf_model
    best_params <- list(ntree = ntree, params = rf_model$bestTune)
  }
}
cat("Best parameters based on validation set :")
print(best_params)
print(best_sensitivity)

# Evaluate tuned model on validation set
rf_tuned_val_pred_cs <- predict(best_model, X_val_rf_cls)
rf_tuned_val_metrics_cs <- confusionMatrix(rf_tuned_val_pred_cs, y_val_cat, positive = "1")
print("Tuned Random Forest - Validation Metrics:")
print(rf_tuned_val_metrics_cs)

# Evaluate tuned model on test set
rf_tuned_test_pred_cs <- predict(best_model, X_test_rf_cls)
rf_tuned_test_metrics_cs <- confusionMatrix(rf_tuned_test_pred_cs, y_test_cat, positive = "1")
print("Tuned Random Forest - Test Metrics:")
print(rf_tuned_test_metrics_cs)
