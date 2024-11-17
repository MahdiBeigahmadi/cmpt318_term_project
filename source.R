required_packages <- c("dplyr", "ggplot2", "depmixS4", "lubridate", "data.table", "devtools", "factoextra", "reshape2", "zoo","car","usethis")

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
library(car)
library(usethis)

# ******************************
#  Read the Dataset 
# ******************************


# Define the file path
file_path <- "Path to dataset"
# Read the data using data.table's fread for efficiency
df <- fread(file_path, header = TRUE, sep = ",", na.strings = "NA", stringsAsFactors = FALSE)

# converting to data.frame if necessary
df <- as.data.frame(df)

# ***********************************
# Inspect the Dataframe Structure
# ***********************************

cat("First 10 rows of the dataframe:\n")
head(df, 10)
cat("Column Names:\n")
colnames(df)


# **************************************************************
# Combining Date and Time into DateTime and Convert to POSIXct
# **************************************************************

df$DateTime <- paste(df$Date, df$Time)
df$DateTime <- as.POSIXct(df$DateTime, format="%d/%m/%Y %H:%M:%S", tz = "UTC")

# verifying the conversion
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
ggplot(df_scaled, aes(x = PC1, y = PC2, color = "orange")) +
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


cat("Training Data Summary (2006-2008):\n")
summary(train_data)
cat("Testing Data Summary (2009):\n")
summary(test_data)

train_features <- train_data[, c("Global_active_power", "Voltage")]
test_features <- test_data[, c("Global_active_power", "Voltage")]

# specifying number of states
states_list <- c(6, 8, 10, 12, 16, 18)
log_likelihoods <- list()
bics <- list()
models <- list()

# ************************************
# Model training
# ************************************
for (num_states in states_list) {
  cat("\nTraining HMM with", num_states, "states...\n")
  
  #  depmix model using gaussian distribution 
  hmm_model <- depmix(
    response = list(Global_active_power ~ 1, Voltage ~ 1),
    data = train_features,
    nstates = num_states,
    family = list(gaussian(), gaussian())
  )
  
  set.seed(123)  
  fitted_model <- fit(hmm_model, verbose = FALSE, emcontrol = em.control(maxit = 5000, tol = 1e-8))
  
  log_likelihood <- logLik(fitted_model)
  bic_value <- BIC(fitted_model)
  
  log_likelihoods[[as.character(num_states)]] <- log_likelihood
  bics[[as.character(num_states)]] <- bic_value
  models[[as.character(num_states)]] <- fitted_model
  
  cat("Log-Likelihood:", log_likelihood, "\n")
  cat("BIC:", bic_value, "\n")
}

# select the best model based on the lowest BIC
best_num_states <- as.numeric(names(which.min(unlist(bics))))
cat("\nBest model has", best_num_states, "states\n")

best_model <- models[[as.character(best_num_states)]]


# Save markov model
saveRDS(best_model, file = "training_model.rds")

# Use for loading in training model.(comment out when not in use).
#training_model_path = "Path to .rds file."
#best_model = readRDS(training_model_path)


# ************************************
# Evaluating performance on test data
# ************************************
test_model <- depmix(
  response = list(Global_active_power ~ 1, Voltage ~ 1),
  data = test_features,
  nstates = best_num_states,
  family = list(gaussian(), gaussian())
)

# using parameters from the best model to predict on test data
test_fitted <- setpars(test_model, getpars(best_model))
test_fitted <- fit(test_fitted, emcontrol = em.control(maxit = 5000, tol = 1e-8))

# extracting log-likelihood for the test data
test_log_likelihood <- logLik(test_fitted)
cat("Log-Likelihood on Test Data:", test_log_likelihood, "\n")






# ************************************
# Display results
# ************************************
cat("\nTraining Results Summary:\n")
result_df <- data.frame(
  States = states_list,
  LogLikelihood = unlist(log_likelihoods),
  BIC = unlist(bics)
)
print(result_df)

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



# ******************************
# Anomaly Detection
# ******************************

# Partition into 10 roughly equal-sized subsets.
# Assign a number to each row of data from 1 to 10.
test_data_partition <- test_data %>%
  mutate(week_group = ntile(row_number(),10))

# Split up the above data into 10 subsets for calculating the log-likelihood.
weekly_subsets <- test_data_partition %>%
  group_split(week_group)

# Computing the log-likelihood for each subset with best performing model.
# Initialize a data frame to store the results
subset_data_frame <- data.frame(
  week_group = 1:10,
  LogLikelihood = numeric(10),
  avg_loglikelihood = numeric(10)
)

# Loop over each subset
for (i in 1:10) {
  subset_data <- weekly_subsets[[i]]
  
  subset_features <- subset_data[, c("Global_active_power", "Voltage")]
  
  #  depmix model using gaussian distribution 
  hmm_model_subset <- depmix(
    response = list(Global_active_power ~ 1, Voltage ~ 1),
    data = subset_features,
    nstates = best_model@nstates,
    family = list(gaussian(), gaussian())
  )
  
  # Set parameters from the best model
  hmm_model_subset <- setpars(hmm_model_subset, getpars(best_model))
  
  # Calculating the log-likelihood
  loglikelihood_subset <- logLik(hmm_model_subset)
  
  ll_per_obs <- loglikelihood_subset/ nrow(subset_features)
  
  # Store the results
  subset_data_frame$LogLikelihood[i] <- loglikelihood_subset
  subset_data_frame$avg_loglikelihood[i] <- ll_per_obs
  
}

# Calculating the results

# Get training log-likelihood per observation
train_log_likelihood <- logLik(best_model) /nrow(train_features)

# Compute deviations from the training log-likelihood per observation
subset_data_frame$Deviation <- subset_data_frame$avg_loglikelihood - train_log_likelihood

# Compute the maximum deviation (threshold value)
threshold <- max(abs(subset_data_frame$Deviation))
cat("Threshold for the acceptable deviation of any unseen observations:", threshold, "\n")

# Display the deviations for each subset
print(subset_data_frame)


















