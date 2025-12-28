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
