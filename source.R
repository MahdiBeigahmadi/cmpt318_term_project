required_packages <- c("dplyr", "ggplot2", "depmixS4", "lubridate", "data.table", "devtools", "factoextra", "reshape2", "zoo","car","usethis","doParallel","foreach")


install_if_missing <- function(packages){
  installed <- rownames(installed.packages())
  for(p in packages){
    if(!(p %in% installed)){
      install.packages(p, dependencies = TRUE)
    }
  }
}

install_if_missing(required_packages)
library(dplyr)
library(ggplot2)
library(lubridate)
library(data.table)
library(usethis)
library(devtools)
library(depmixS4)  
library(factoextra)  
library(reshape2)    
library(zoo)         
library("car")
library(doParallel)
library(foreach)

file_path <- "The location of the txt file goes here"

df <- fread(file_path, header = TRUE, sep = ",", na.strings = "NA", stringsAsFactors = FALSE)

# Converting to data.frame if necessary
df <- as.data.frame(df)

cat("First 10 rows of the dataframe:\n")
head(df, 10)
cat("Column Names:\n")
colnames(df)

# **************************************************************
# Combining Date and Time into DateTime and Convert to POSIXct
# **************************************************************

df$DateTime <- paste(df$Date, df$Time)
df$DateTime <- as.POSIXct(df$DateTime, format="%d/%m/%Y %H:%M:%S", tz = "UTC")

cat("DateTime conversion completed.\n")
str(df$DateTime)

# ******************************
# Extract Time Window on Monday (09:00 AM to 12:00 PM)
# ******************************

# Define the function to extract the time window
extract_time_window <- function(dataframe) {
  df_monday_9am_to_12pm <- dataframe %>%
    filter(weekdays(DateTime) == "Monday" &
             hour(DateTime) >= 9 & hour(DateTime) < 12)
  return(df_monday_9am_to_12pm)
}

# Now, apply the function to the dataframe
# df <- extract_time_window(df)

# View the extracted time window data
#cat("Extracted Time Window Data (09:00 AM to 12:00 PM on Monday):\n")
print(head(df))

# ******************************
# Convert Columns to Numeric
# ******************************

# Convert numeric columns to numeric type
numeric_cols <- c("Global_active_power", "Global_reactive_power", "Voltage", 
                  "Global_intensity", "Sub_metering_1", "Sub_metering_2", 
                  "Sub_metering_3")

df[numeric_cols] <- lapply(df[numeric_cols], function(x) as.numeric(x))

# Checking for conversion success
if(any(sapply(df[numeric_cols], function(x) any(is.na(x))))){
  cat("Warning: Some numeric columns have NA values after conversion.\n")
} else {
  cat("All numeric columns converted successfully.\n")
}

# ******************************
# Handling the Missing Values
# ******************************

# Check for missing values
missing_values <- sapply(df[numeric_cols], function(x) sum(is.na(x)))
cat("Missing Values in Each Numeric Column:\n")
print(missing_values)

# Approximating the NA values
fill_na <- function(x) {
  # For interpolation
  x <- na.approx(x, na.rm = FALSE)
  # To handle leading NAs
  x <- na.locf(x, na.rm = FALSE)
  # To handle trailing NAs
  x <- na.locf(x, na.rm = FALSE, fromLast = TRUE)
  return(x)
}

df[numeric_cols] <- lapply(df[numeric_cols], fill_na)

missing_values_after <- sapply(df[numeric_cols], function(x) sum(is.na(x)))
cat("Missing Values in Each Numeric Column (After Interpolation):\n")
print(missing_values_after)

df_clean <- df

# ******************************
# Feature Engineering
# ******************************

# Creating new features based on domain knowledge

# Total Sub Metering
# df_clean$Total_sub_metering <- df_clean$Sub_metering_1 + df_clean$Sub_metering_2 + df_clean$Sub_metering_3

# Time-based features
df_clean$Hour <- as.integer(format(df_clean$DateTime, "%H"))
df_clean$DayOfWeek <- as.factor(weekdays(df_clean$DateTime))
df_clean$Month <- as.factor(format(df_clean$DateTime, "%m"))

# Removing initial rows with NA due to lag and rolling calculations
df_clean <- df_clean[complete.cases(df_clean), ]

# Update numeric columns to include new features
numeric_cols <- c('Global_active_power', 'Global_reactive_power', 'Voltage','Global_intensity',
                  'Sub_metering_1','Sub_metering_2','Sub_metering_3')

# ************************************
# Feature Scaling (Standardization)
# ************************************

df_scaled <- df_clean
df_scaled[numeric_cols] <- scale(df_scaled[numeric_cols])

# Check the scaling results making sure Mean = 0
cat("Summary of Scaled Variables:\n")
print(summary(df_scaled[numeric_cols]))

# Making sure SD = 1 
col_sds <- sapply(df_scaled[numeric_cols], sd, na.rm = TRUE)

# Display the standard deviations
cat("Standard Deviations of All Columns:\n")
print(col_sds)

# ************************************
# Principal Component Analysis (PCA)
# ************************************

# preparing data for PCA
pca_data <- df_scaled[numeric_cols]

# performing PCA
pca_result <- prcomp(pca_data, center = FALSE, scale. = FALSE)

# Summary of PCA results
cat("PCA Summary:\n")
print(summary(pca_result))

# variance percentages of all PCs
pca_var <- pca_result$sdev^2
pca_var_perc <- pca_var / sum(pca_var) * 100

# Print variance percentages of all PCs
cat("Variance percentages of all Principal Components:\n")
for (i in 1:length(pca_var_perc)) {
  cat(paste0("PC", i, ": ", round(pca_var_perc[i], 2), "% "))
}

fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 50)) +
  labs(title = "Variance Percentage vs Principal Component",
       x = "Principal Components",
       y = "Percentage of Variance Explained")

# Add PCA scores to the dataframe
df_scaled$PC1 <- pca_result$x[,1]
df_scaled$PC2 <- pca_result$x[,2]

# ************************************
# Visualizations with PCA Components
# ***********************************

# Correlation Plot of Original Variables
cor_matrix <- cor(df_scaled[, numeric_cols])

melted_cor <- melt(cor_matrix)

ggplot(data = melted_cor, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Correlation") +
  theme_minimal() +
  labs(title = "Correlation Matrix of Variables", x = "", y = "") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 10, hjust = 1))

# Extracting the loadings (rotation matrix) from the PCA results
loadings <- pca_result$rotation

# scaling the loadings for better visualization 
scale_factor <- 5
loadings_scaled <- loadings[, 1:2] * scale_factor

# preparing a data frame for the loadings (arrows)
arrow_data <- data.frame(
  Feature = rownames(loadings_scaled),
  PC1 = loadings_scaled[, 1],
  PC2 = loadings_scaled[, 2]
)

# PCA plot PC1 vs PC2
ggplot(df_scaled, aes(x = PC1, y = PC2, color = "green2")) +
  geom_jitter(alpha = 0.5, size = 2, width = 0.2, height = 0.2) + 
  scale_color_brewer() +
  geom_segment(data = arrow_data, aes(x = 0, y = 0, xend = PC1, yend = PC2), 
               arrow = arrow(type = "closed", length = unit(0.2, "cm")), 
               color = "blue", linewidth = 1) +  
  geom_text(data = arrow_data, aes(x = PC1, y = PC2, label = Feature), 
            hjust = 1.2, vjust = 1.2, size = 5, color = "black") + 
  labs(
    title = "PCA Scatter Plot: Feature Contribution",
    x = "Principal Component 1",
    y = "Principal Component 2",
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16),
    axis.title = element_text(size = 14),
  )

pca_result <- prcomp(df_scaled[numeric_cols])
loadings <- pca_result$rotation
print(loadings)

df_scaled$Year <- year(df_scaled$DateTime)

print(pca_result$rotation[2,0])

# PC1 scores for each variable
loading_scores <- pca_result$rotation[, 1]
# top two features magnitude (highest ones)
most_influential_features <- names(sort(abs(loading_scores), decreasing = TRUE)[1:2])

print(most_influential_features)


# ************************************
# Splitting train and test data
# ************************************
df_scaled <- extract_time_window(df_scaled)

train_data <- df_scaled %>% filter(Year <= 2008)
test_data <- df_scaled %>% filter(Year == 2009)

train_features <- train_data[, c("Global_intensity","Voltage")]
test_features <- test_data[, c("Global_intensity","Voltage")]

# ************************************
# Model Training Optimizations
# ************************************

# Reduce the number of states to try
states_list <- c(4, 6, 7, 8, 10, 12, 13)

# Adjust EM algorithm control parameters
em_ctrl <- em.control(maxit = 1000, tol = 1e-5)

# Initialize lists to store results
log_likelihoods <- list()
bics <- list()
models <- list()

# Parallelize model training (requires doParallel and foreach packages)

# Set up parallel backend to use multiple processors
num_cores <- detectCores()   # Leave one core free
cl <- makeCluster(num_cores)
registerDoParallel(cl)


results <- foreach(num_states = states_list, .packages = 'depmixS4') %dopar% {
  suppressMessages({
    hmm_model <- depmix(
      response = list( Global_intensity ~ 1, Voltage ~ 1),
      data = train_features,
      nstates = num_states,
      family = list(gaussian(), gaussian())
    )
    
    set.seed(42)
    print(paste0("train model state = ", num_states))
    fitted_model <- fit(hmm_model, ntimes = 10, verbose = FALSE, emcontrol = em_ctrl)
    
    log_likelihood <- logLik(fitted_model)
    bic_value <- BIC(fitted_model)
    
    list(
      num_states = num_states,
      log_likelihood = log_likelihood,
      bic_value = bic_value,
      model = fitted_model
    )
  })
}

stopCluster(cl)

# Collect results from the parallel computations
for (res in results) {
  num_states <- res$num_states
  log_likelihoods[[as.character(num_states)]] <- res$log_likelihood
  bics[[as.character(num_states)]] <- res$bic_value
  models[[as.character(num_states)]] <- res$model
  cat("Log-Likelihood for", num_states, "states:", res$log_likelihood, "\n")
  cat("BIC for", num_states, "states:", res$bic_value, "\n")
}

best_num_states = 7
best_model <- models[[as.character(best_num_states)]]

cat("The Best Model has", best_num_states, "states\n")
print(best_model)

# Save the best model
saveRDS(best_model, file = "training_model.rds")

cat("\nTraining Results Summary:\n")
result_df <- data.frame(
  States = states_list,
  LogLikelihood = unlist(log_likelihoods),
  BIC = unlist(bics)
)
print(result_df)

# Plotting BIC and log-likelihood
ggplot(result_df, aes(x = States)) +
  geom_line(aes(y = BIC, color = "BIC"), linewidth = 1) +
  geom_point(aes(y = BIC, color = "BIC"), size = 3) +
  
  geom_line(aes(y = LogLikelihood, color = "Log-Likelihood"), linewidth = 1) +
  geom_point(aes(y = LogLikelihood, color = "Log-Likelihood"), size = 3) +
  labs(
    title = "BIC and Log Likelihood for Different Number of States",
    x = "Number of States",
    y = "Value",
    color = "Metric"
  ) +
  scale_color_manual(values = c("BIC" = "blue", "Log-Likelihood" = "red")) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14),
    legend.position = "top",
    axis.title.y = element_text(size = 12),
    axis.title.x = element_text(size = 12)
  )

# Plot of BIC vs Number of States
ggplot(result_df, aes(x = States, y = BIC)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_point(color = "red", size = 3) +
  labs(title = "BIC vs. Number of States",
       x = "Number of States",
       y = "BIC Value") +
  theme_minimal()

# Plot of Log-Likelihood vs Number of States
ggplot(result_df, aes(x = States, y = LogLikelihood)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_point(color = "red", size = 3) +
  labs(title = "Log-Likelihood vs. Number of States",
       x = "Number of States",
       y = "Log-Likelihood") +
  theme_minimal()


# ************************************
# Evaluating Performance on Test Data
# ************************************

# Using the best model parameters to predict on test data
test_model <- depmix(
  response = list(Global_intensity ~ 1, Voltage ~ 1),
  data = test_features,
  nstates = best_num_states,
  family = list(gaussian(), gaussian())
)

# Setting parameters from the best model
test_fitted <- setpars(test_model, getpars(best_model))

# Compute the log-likelihood without re-fitting
fb_test <- forwardbackward(test_fitted)
test_log_likelihood <- fb_test$logLike
cat("Log-Likelihood on Test Data:", test_log_likelihood, "\n")


# ******************************
# Anomaly Detection Optimization
# ******************************

# Partition into 10 roughly equal-sized subsets
test_data_partition <- test_data %>%
  mutate(week_group = ntile(row_number(), 10))

weekly_subsets <- test_data_partition %>%
  group_split(week_group)

# Initialize data frame to store results
subset_data_frame <- data.frame(
  week_group = 1:10,
  LogLikelihood = numeric(10),
  avg_loglikelihood = numeric(10)
)

# Anomaly detection loop without using setdata
for (i in 1:10) {
  subset_data <- weekly_subsets[[i]]
  subset_features <- subset_data[, c("Global_intensity", "Voltage")]
  
  # Create a new model with the subset data
  hmm_model_subset <- depmix(
    response = list(Global_intensity ~ 1, Voltage ~ 1),
    data = subset_features,
    nstates = best_num_states,
    family = list(gaussian(), gaussian())
  )
  
  # choosing the parameters from the best model
  hmm_model_subset <- setpars(hmm_model_subset, getpars(best_model))
  
  # computing the log-likelihood without re-fitting
  fb <- forwardbackward(hmm_model_subset)
  loglikelihood_subset <- fb$logLike
  normalize_loglikelihood_subset <- loglikelihood_subset / nrow(subset_features)
  
  subset_data_frame$LogLikelihood[i] <- loglikelihood_subset
  subset_data_frame$avg_loglikelihood[i] <- normalize_loglikelihood_subset
}

# calculating deviations and threshold
#train_log_likelihood <- logLik(best_model) / nrow(train_features)
train_log_likelihood <- forwardbackward(best_model)$logLike / nrow(train_features)
subset_data_frame$Deviation <- subset_data_frame$avg_loglikelihood - train_log_likelihood
threshold <- max(abs(subset_data_frame$Deviation))
cat("Threshold for the acceptable deviation of any unseen observations:", threshold, "\n")
print(subset_data_frame)

# ************************************
#  Log-Likelihood for Training Data
# ************************************

# the best model fitted on the training dataset
train_fitted <- setpars(models[[as.character(best_num_states)]], getpars(models[[as.character(best_num_states)]]))

# log-likelihood for training data using the forward-backward algorithm
fb_train <- forwardbackward(train_fitted)
train_log_likelihood <- fb_train$logLik

cat("Log-Likelihood for Training Data: ", train_log_likelihood, "\n")


# ************************************
#  Log-Likelihood for Test Data
# ************************************

# the best model fitted on the test dataset
test_fitted <- setpars(test_model, getpars(models[[as.character(best_num_states)]]))

#  log-likelihood for test data using the forward-backward algorithm
fb_test <- forwardbackward(test_fitted)
test_log_likelihood <- fb_test$logLik

cat("Log-Likelihood for Test Data: ", test_log_likelihood, "\n")

# ************************************
#  Normalized Log-Likelihood
# ************************************

# nomalizing the log-likelihood by dividing by the number of observations
train_log_likelihood_normalized <- train_log_likelihood / nrow(train_data)
test_log_likelihood_normalized <- test_log_likelihood / nrow(test_data)

cat("Normalized Log-Likelihood for Training Data: ", train_log_likelihood_normalized, "\n")
cat("Normalized Log-Likelihood for Test Data: ", test_log_likelihood_normalized, "\n")

# ************************************
# Comparison plot 
# ************************************

comparison_df <- data.frame(
  Data = c("Training", "Test"),
  LogLikelihood = c(train_log_likelihood_normalized, test_log_likelihood_normalized)
)

ggplot(comparison_df, aes(x = Data, y = LogLikelihood, fill = Data)) +
  geom_bar(stat = "identity") +
  labs(title = "Normalized Log-Likelihood Comparison: Training vs Test",
       x = "Dataset", y = "Normalized Log-Likelihood") +
  theme_minimal() +
  theme(legend.position = "none")

