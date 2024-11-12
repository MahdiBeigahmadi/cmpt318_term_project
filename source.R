required_packages <- c("dplyr", "ggplot2", "depmixS4", "lubridate", "data.table", "devtools", "factoextra", "reshape2", "zoo")
install.packages("car")

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
library(devtools)
library(depmixS4)  # Load depmixS4 after other packages
library(factoextra)  # For PCA visualization
library(reshape2)    # For correlation plot
library(zoo)         # For rolling mean calculation
library("car")

# ******************************
#  Read the Dataset 
# ******************************


# Define the file path
file_path <- "/Users/koushaamouzesh/Desktop/Fall 2024/318/term project/group_project/TermProjectData.txt"

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
    filter(weekdays(DateTime) == "Monday" & hour(DateTime) >= 9 & hour(DateTime) < 12)
  
  
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

# lag feature: Lag of Global Active Power
df_clean$Global_active_power_lag1 <- dplyr::lag(df_clean$Global_active_power, n = 1)

# rolling average: 24-hour Rolling Mean of Global Active Power
df_clean$Global_active_power_rollmean <- zoo::rollapply(df_clean$Global_active_power, width = 24, FUN = mean, align = "right", fill = NA)

# removing initial rows with NA due to lag and rolling calculations
df_clean <- df_clean[complete.cases(df_clean), ]

# Update numeric columns to include new features

numeric_cols <- c('Global_active_power', 'Global_reactive_power', 'Voltage','Global_intensity',
                  'Sub_metering_1','Sub_metering_2','Sub_metering_3', 'Total_sub_metering',
                  'Global_active_power_lag1', 'Global_active_power_rollmean')

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

# Scree plot to determine the number of components
fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 50)) +
  labs(title = "Scree Plot",
       x = "Principal Components",
       y = "Percentage of Variance Explained")

# Biplot of the first two principal components
fviz_pca_biplot(pca_result, geom = "point", 
                addEllipses = TRUE, ellipse.level = 0.95, 
                palette = "jco") +
  labs(title = "PCA Biplot Colored by Day of Week") +
  theme_minimal()

# add PCA scores to the dataframe
df_scaled$PC1 <- pca_result$x[,1]
df_scaled$PC2 <- pca_result$x[,2]

# ************************************
# Visualizations with PCA Components
# ************************************

# Time series plot of PC1
ggplot(df_scaled, aes(x = DateTime, y = PC1)) +
  geom_line(color = "steelblue") +
  labs(title = "Time Series of First Principal Component", x = "Time", y = "PC1") +
  theme_minimal()

# Scatter plot of PC1 vs PC2 colored by Day of Week
ggplot(df_scaled, aes(x = PC1, y = PC2)) +
  geom_point(alpha = 0.2) +
  labs(title = "PCA Scatter Plot by Day of Week",
       x = "Principal Component 1", y = "Principal Component 2") +
  theme_minimal()


# Box Plot of PC1 by Month
ggplot(df_scaled, aes(x = Month, y = PC1)) +
  geom_boxplot(fill = "lightgreen") +
  labs(title = "Distribution of PC1 by Month", x = "Month", y = "PC1") +
  theme_minimal()

# Density Plot of PC1
ggplot(df_scaled, aes(x = PC1)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot of PC1 by Day of Week", x = "PC1", y = "Density") +
  theme_minimal()

# Correlation Plot of Original Variables
cor_matrix <- cor(df_scaled[, numeric_cols])

# Melt the correlation matrix for plotting
melted_cor <- melt(cor_matrix)

# Plot the correlation matrix
ggplot(data = melted_cor, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Correlation") +
  theme_minimal() +
  labs(title = "Correlation Matrix of Variables", x = "", y = "") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 10, hjust = 1))

