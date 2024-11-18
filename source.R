required_packages <- c("dplyr", "ggplot2", "depmixS4", "lubridate", "data.table", "devtools", "factoextra", "reshape2", "zoo")
install.packages("car")
install.packages('usethis')
install.packages('doParallel')
install.packages("foreach")


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

file_path <- "/Users/koushaamouzesh/Desktop/Fall 2024/318/term project/group_project/TermProjectData.txt"

df <- fread(file_path, header = TRUE, sep = ",", na.strings = "NA", stringsAsFactors = FALSE)

# converting to data.frame if necessary
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
# Extract Time Window on Monday (09:00 AM to 12:00 AM)
# ******************************

# Define the function to extract the time window
extract_time_window <- function(dataframe) {
  df_monday_9am_to_12pm <- dataframe %>%
    filter(weekdays(DateTime) == "Wednesday" & hour(DateTime) >= 9 & hour(DateTime) < 12)
  
  
  return(df_monday_9am_to_12pm)
}

# Now, apply the function to the dataframe
df <- extract_time_window(df)

# View the extracted time window data
cat("Extracted Time Window Data (09:00 AM to 12:00 AM on Monday):\n")
print(head(df))

# ******************************
# Convert Columns to Numeric
# ******************************

# Convert numeric columns to numeric type
numeric_cols <- c("Global_active_power", "Global_reactive_power", "Voltage", 
                  "Global_intensity", "Sub_metering_1", "Sub_metering_2", 
                  "Sub_metering_3")

df[numeric_cols] <- lapply(df[numeric_cols], function(x) as.numeric(x))

# checking for conversion success
if(any(sapply(df[numeric_cols], function(x) any(is.na(x))))){
  cat("Warning: Some numeric columns have NA values after conversion.\n")
} else {
  cat("All numeric columns converted successfully.\n")
}

# ******************************
# Handling the Missing Values
# ******************************

# check for missing values
missing_values <- sapply(df[numeric_cols], function(x) sum(is.na(x)))
cat("Missing Values in Each Numeric Column:\n")
print(missing_values)

# aproximating the NA values
fill_na <- function(x) {
  # for interpolation
  x <- na.approx(x, na.rm = FALSE)
  # to handle leading NAs
  x <- na.locf(x, na.rm = FALSE)
  # to handle trailing NAs
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

# creating new features based on domain knowledge

# total Sub Metering
df_clean$Total_sub_metering <- df_clean$Sub_metering_1 + df_clean$Sub_metering_2 + df_clean$Sub_metering_3

# time-based features
df_clean$Hour <- as.integer(format(df_clean$DateTime, "%H"))
df_clean$DayOfWeek <- as.factor(weekdays(df_clean$DateTime))
df_clean$Month <- as.factor(format(df_clean$DateTime, "%m"))

# removing initial rows with NA due to lag and rolling calculations
df_clean <- df_clean[complete.cases(df_clean), ]

# Update numeric columns to include new features

numeric_cols <- c('Global_active_power', 'Global_reactive_power', 'Voltage','Global_intensity',
                  'Sub_metering_1','Sub_metering_2','Sub_metering_3', 'Total_sub_metering')

# ************************************
# Feature Scaling (Standardization)
# ************************************

df_scaled <- df_clean
df_scaled[numeric_cols] <- scale(df_scaled[numeric_cols])

# check the scaling results making sure Mean = 0
cat("Summary of Scaled Variables:\n")
print(summary(df_scaled[numeric_cols]))

# making sure SD = 1 
col_sds <- sapply(df_scaled[numeric_cols], sd, na.rm = TRUE)

# Display the standard deviations
cat("Standard Deviations of All Columns:\n")
print(col_sds)



# ************************************
# Principal Component Analysis (PCA)
# ************************************

# Prepare data for PCA
pca_data <- df_scaled[numeric_cols]

# Perform PCA
pca_result <- prcomp(pca_data, center = FALSE, scale. = FALSE)

# Summary of PCA results
cat("PCA Summary:\n")
print(summary(pca_result))

# Variance percentages of all PCs
pca_var <- pca_result$sdev^2
pca_var_perc <- pca_var / sum(pca_var) * 100

# Print variance percentages of all PCs
cat("Variance percentages of all Principal Components:\n")
for (i in 1:length(pca_var_perc)) {
  cat(paste0("PC", i, ": ", round(pca_var_perc[i], 2), "%\n"))
}


# add PCA scores to the dataframe
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


# extracting the loadings (rotation matrix) from the PCA results
# the values in the matrix show the contributions of each variable
# to the principal components
loadings <- pca_result$rotation

# scaling the loadings for better visualization 
scale_factor <- 5
loadings_scaled <- loadings[, 1:2] * scale_factor

# Prepare a data frame for the loadings (arrows)
arrow_data <- data.frame(
  Feature = rownames(loadings_scaled),
  PC1 = loadings_scaled[, 1],
  PC2 = loadings_scaled[, 2]
)

# PCA plot PC1 vs PC2
ggplot(df_scaled, aes(x = PC1, y = PC2, color = "green")) +
  geom_jitter(alpha = 0.5, size = 2, width = 0.2, height = 0.2) +  # Jitter to reduce overlap
  scale_color_brewer(palette = "Dark2") +
  geom_segment(data = arrow_data, aes(x = 0, y = 0, xend = PC1, yend = PC2), 
               arrow = arrow(type = "closed", length = unit(0.2, "cm")), 
               color = "blue", size = 1) +  
  geom_text(data = arrow_data, aes(x = PC1, y = PC2, label = Feature), 
            hjust = 1.2, vjust = 1.2, size = 4, color = "darkred") + 
  labs(
    title = "Simplified PCA Scatter Plot with Feature Arrows",
    x = "Principal Component 1",
    y = "Principal Component 2",
    color = "Day of the Week"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16),
    axis.title = element_text(size = 14),
    legend.position = "bottom"
  )

pca_result <- prcomp(df_scaled[numeric_cols])
loadings <- pca_result$rotation
print(loadings)



df_scaled$Year <- year(df_scaled$DateTime)

# ************************************
# Splitting train and test data
# ************************************
                               
train_data <- df_scaled %>% filter(Year <= 2008)
test_data <- df_scaled %>% filter(Year == 2009)

train_features <- train_data[, c("Global_intensity","Voltage")]
test_features <- test_data[, c("Global_intensity","Voltage")]

# ************************************
# Model Training Optimizations
# ************************************

# 1. Reduce the number of states to try
states_list <- c(4, 6, 8, 10, 12, 16)

# 2. Adjust EM algorithm control parameters
em_ctrl <- em.control(maxit = 1000, tol = 1e-5)

# Optional: 3. Sample 50% of the training data to reduce size
# Comment this out if you want to use all data
# train_features <- train_features %>% sample_frac(0.5)

# Initialize lists to store results
log_likelihoods <- list()
bics <- list()
models <- list()

# 4. Parallelize model training (requires doParallel and foreach packages)

# Set up parallel backend to use multiple processors
num_cores <- detectCores() - 1  # Leave one core free
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Parallelized model training loop
results <- foreach(num_states = states_list, .packages = 'depmixS4') %dopar% {
  # Suppress output in parallel processing
  suppressMessages({
    hmm_model <- depmix(
      response = list(Global_intensity ~ 1, Voltage ~ 1),
      data = train_features,
      nstates = num_states,
      family = list(gaussian(), gaussian())
    )
    
    set.seed(42)
    print(paste0("train model state = ", num_states))
    fitted_model <- fit(hmm_model, verbose = FALSE, emcontrol = em_ctrl)
    
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

# Stop the cluster after computations
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

# Select the best model based on the lowest BIC
best_num_states <- as.numeric(names(which.min(unlist(bics))))
cat("\nBest model has", best_num_states, "states\n")

best_model <- models[[as.character(best_num_states)]]

# Save the best model
saveRDS(best_model, file = "training_model.rds")


cat("\nTraining Results Summary:\n")
result_df <- data.frame(
  States = states_list,
  LogLikelihood = unlist(log_likelihoods),
  BIC = unlist(bics)
)
print(result_df)

# plotting BIC and log-liklihood
ggplot(result_df, aes(x = States)) +
  geom_line(aes(y = BIC, color = "BIC"), size = 1) +
  geom_point(aes(y = BIC, color = "BIC"), size = 3) +
  
  geom_line(aes(y = LogLikelihood, color = "Log-Likelihood"), size = 1) +
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


# ************************************
# Evaluating Performance on Test Data
# ************************************

# using the best model parameters to predict on test data
test_model <- depmix(
  response = list(Global_intensity ~ 1, Voltage ~ 1),
  data = test_features,
  nstates = best_num_states,
  family = list(gaussian(), gaussian())
)

# setting parameters from the best model
test_fitted <- setpars(test_model, getpars(best_model))
test_fitted <- fit(test_fitted, emcontrol = em_ctrl, verbose = FALSE)

# extracting log-likelihood for the test data
test_log_likelihood <- logLik(test_fitted)
cat("Log-Likelihood on Test Data:", test_log_likelihood, "\n")

# ************************************
# Display Training Results
# ************************************

# Plot BIC vs Number of States
ggplot(result_df, aes(x = States, y = BIC)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "red", size = 3) +
  labs(title = "BIC vs. Number of States",
       x = "Number of States",
       y = "BIC Value") +
  theme_minimal()

# Plot Log-Likelihood vs Number of States
ggplot(result_df, aes(x = States, y = LogLikelihood)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "red", size = 3) +
  labs(title = "Log-Likelihood vs. Number of States",
       x = "Number of States",
       y = "Log-Likelihood") +
  theme_minimal()

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

# Optimize anomaly detection loop
for (i in 1:10) { # HAS BUG NEEDS TO BE FIXED!!!!!!!!!
  subset_data <- weekly_subsets[[i]]
  
  subset_features <- subset_data[, c("Global_intensity", "Voltage")]
  
  hmm_model_subset <- depmix(
    response = list(Global_intensity ~ 1, Voltage ~ 1),
    data = subset_features,
    nstates = best_num_states,
    family = list(gaussian(), gaussian())
  )
  
  # Set parameters from the best model
  hmm_model_subset <- setpars(hmm_model_subset, getpars(best_model))
  
  # Fit the model with fewer iterations and higher tolerance
  hmm_model_subset <- fit(hmm_model_subset, emcontrol = em_ctrl, verbose = FALSE)
  
  # Calculate log-likelihood
  loglikelihood_subset <- logLik(hmm_model_subset)
  ll_per_obs <- loglikelihood_subset / nrow(subset_features)
  
  # Store the results
  subset_data_frame$LogLikelihood[i] <- loglikelihood_subset
  subset_data_frame$avg_loglikelihood[i] <- ll_per_obs
}

# Calculate deviations and threshold
train_log_likelihood <- logLik(best_model) / nrow(train_features)
subset_data_frame$Deviation <- subset_data_frame$avg_loglikelihood - train_log_likelihood
threshold <- max(abs(subset_data_frame$Deviation))
cat("Threshold for the acceptable deviation of any unseen observations:", threshold, "\n")
print(subset_data_frame)




# ************************************
# Display results
# ************************************

# plot of BIC vs # of states
ggplot(result_df, aes(x = States, y = BIC)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "red", size = 3) +
  labs(title = "BIC vs. Number of States",
       x = "Number of States",
       y = "BIC Value") +
  theme_minimal()


ggplot(result_df, aes(x = States, y = LogLikelihood)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "red", size = 3) +
  labs(title = "Log-liklihood vs. Number of States",
       x = "Number of States",
       y = "Log-liklihood") +
  theme_minimal()


ggplot(result_df, aes(x = States)) +
  geom_line(aes(y = BIC, color = "BIC"), size = 1) +
  geom_point(aes(y = BIC, color = "BIC"), size = 3) +
  
  geom_line(aes(y = LogLikelihood, color = "Log-Likelihood"), size = 1) +
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

