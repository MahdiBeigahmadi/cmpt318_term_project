required_packages <- c("dplyr", "ggplot2", "depmixS4", "lubridate", "data.table", "devtools", "factoextra", "reshape2", "zoo")

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

# ******************************
# 2. Read the Dataset Correctly
# ******************************

# Define the file path
file_path <- "/Users/mahdi/Desktop/termproject/TermProjectData.txt"

# Read a few lines to determine the separator
sample_lines <- readLines(file_path, n = 5)

# Function to detect separator based on the first line
detect_separator <- function(line){
  if(grepl(",", line)){
    return(",")
  } else if(grepl(";", line)){
    return(";")
  } else if(grepl("\t", line)){
    return("\t")
  } else {
    stop("Unable to determine the delimiter. Please specify the correct separator.")
  }
}

# Detect separator
separator <- detect_separator(sample_lines[1])
cat("Detected separator:", ifelse(separator == "\t", "Tab", ifelse(separator == ";", "Semicolon", "Comma")), "\n")

# Read the data using data.table's fread for efficiency
df <- fread(file_path, header = TRUE, sep = separator, na.strings = "NA", stringsAsFactors = FALSE)

# Convert to data.frame if necessary
df <- as.data.frame(df)

# ***********************************
# 3. Inspect the Dataframe Structure
# ***********************************

# Inspect the structure of the dataframe
cat("Structure of the dataframe:\n")
str(df)

# View the first few rows
cat("First 10 rows of the dataframe:\n")
print(head(df, 10))

# Print column names
cat("Column Names:\n")
print(colnames(df))

# ******************************
# 4. Rename Columns if Necessary
# *******************************

# Define expected column names
expected_columns <- c("Date", "Time", "Global_active_power", "Global_reactive_power", 
                      "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", 
                      "Sub_metering_3")

# Check if the column names match expected
if(!all(expected_columns %in% colnames(df))){
  # Attempt to rename columns by replacing dots with underscores
  colnames(df) <- gsub("\\.", "_", colnames(df))
  
  # Re-check if renaming fixed the issue
  if(all(expected_columns %in% colnames(df))){
    cat("Columns renamed to expected names.\n")
  } else {
    # Identify which columns are still missing
    missing_cols_after_rename <- setdiff(expected_columns, colnames(df))
    stop(paste("After renaming, the following expected columns are still missing:", 
               paste(missing_cols_after_rename, collapse = ", ")))
  }
} else {
  cat("All expected columns are present.\n")
}

# ********************************************
# 5. Verify Column Existence and Correctness
# ********************************************

# Define expected numeric columns
numeric_cols <- c("Global_active_power", "Global_reactive_power", "Voltage", 
                  "Global_intensity", "Sub_metering_1", "Sub_metering_2", 
                  "Sub_metering_3")

# Check if all expected numeric columns exist
missing_cols <- setdiff(numeric_cols, colnames(df))
if(length(missing_cols) > 0){
  stop(paste("The following expected columns are missing in the data:", paste(missing_cols, collapse = ", ")))
} else {
  cat("All expected numeric columns are present.\n")
}

# Check for duplicate column names
duplicated_cols <- colnames(df)[duplicated(colnames(df))]
if(length(duplicated_cols) > 0){
  stop(paste("Duplicate column names found:", paste(duplicated_cols, collapse = ", ")))
} else {
  cat("No duplicate column names found.\n")
}

# **************************************************************
# 6. Combine Date and Time into DateTime and Convert to POSIXct
# **************************************************************

# Combine Date and Time into DateTime
df$DateTime <- paste(df$Date, df$Time)

# Convert DateTime to POSIXct format
df$DateTime <- as.POSIXct(df$DateTime, format="%d/%m/%Y %H:%M:%S", tz = "UTC")

# Verify the conversion
cat("DateTime conversion completed.\n")
str(df$DateTime)

# ******************************
# 7. Convert Columns to Numeric
# ******************************

# Convert numeric columns to numeric type
df[numeric_cols] <- lapply(df[numeric_cols], function(x) as.numeric(x))

# Check for conversion success
if(any(sapply(df[numeric_cols], function(x) any(is.na(x))))){
  cat("Warning: Some numeric columns have NA values after conversion.\n")
} else {
  cat("All numeric columns converted successfully.\n")
}

# ******************************
# 8. Handling the Missing Values
# ******************************

# Check for missing values
missing_values <- sapply(df[numeric_cols], function(x) sum(is.na(x)))
cat("Missing Values in Each Numeric Column:\n")
print(missing_values)

# Option 1: Remove rows with any missing values
df_clean <- df[complete.cases(df), ]
cat("Rows with missing values removed. Cleaned dataframe has", nrow(df_clean), "rows.\n")

# ******************************
# 9. Feature Engineering
# ******************************

# Create new features based on domain knowledge

# Total Sub Metering
df_clean$Total_sub_metering <- df_clean$Sub_metering_1 + df_clean$Sub_metering_2 + df_clean$Sub_metering_3

# Time-based features
df_clean$Hour <- as.integer(format(df_clean$DateTime, "%H"))
df_clean$DayOfWeek <- as.factor(weekdays(df_clean$DateTime))
df_clean$Month <- as.factor(format(df_clean$DateTime, "%m"))

# Lag Feature: Lag of Global Active Power
df_clean$Global_active_power_lag1 <- dplyr::lag(df_clean$Global_active_power, n = 1)

# Rolling Average: 24-hour Rolling Mean of Global Active Power
df_clean$Global_active_power_rollmean <- zoo::rollapply(df_clean$Global_active_power, width = 24, FUN = mean, align = "right", fill = NA)

# Remove initial rows with NA due to lag and rolling calculations
df_clean <- df_clean[complete.cases(df_clean), ]

# Update numeric columns to include new features
numeric_cols <- c(numeric_cols, "Total_sub_metering", "Hour", "Global_active_power_lag1", "Global_active_power_rollmean")

# ************************************
# 10. Feature Scaling (Standardization)
# ************************************

# Standardize the numeric variables (mean = 0, sd = 1)
scaled_vars <- scale(df_clean[numeric_cols])

# Check the scaling results
cat("Summary of Scaled Variables:\n")
print(summary(scaled_vars))

# Combine scaled variables back into the dataframe
df_scaled <- df_clean %>%
  dplyr::select(DateTime, DayOfWeek, Month) %>%
  bind_cols(as.data.frame(scaled_vars))

# Verify the combined dataframe
str(df_scaled)

# ************************************
# 11. Principal Component Analysis (PCA)
# ************************************

# Prepare data for PCA
pca_data <- df_scaled %>% dplyr::select(-DateTime, -DayOfWeek, -Month)

# Perform PCA
pca_result <- prcomp(pca_data, center = FALSE, scale. = FALSE)  # Data is already scaled

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
fviz_pca_biplot(pca_result, geom = "point", habillage = df_scaled$DayOfWeek, 
                addEllipses = TRUE, ellipse.level = 0.95, 
                palette = "jco") +
  labs(title = "PCA Biplot Colored by Day of Week") +
  theme_minimal()

# Add PCA scores to the dataframe
df_scaled$PC1 <- pca_result$x[,1]
df_scaled$PC2 <- pca_result$x[,2]

# ************************************
# 12. Visualizations with PCA Components
# ************************************

# Time series plot of PC1
ggplot(df_scaled, aes(x = DateTime, y = PC1)) +
  geom_line(color = "steelblue") +
  labs(title = "Time Series of First Principal Component", x = "Time", y = "PC1") +
  theme_minimal()

# Scatter plot of PC1 vs PC2 colored by Day of Week
ggplot(df_scaled, aes(x = PC1, y = PC2, color = DayOfWeek)) +
  geom_point(alpha = 0.6) +
  labs(title = "PCA Scatter Plot by Day of Week",
       x = "Principal Component 1", y = "Principal Component 2") +
  theme_minimal()

# Box Plot of PC1 by Month
ggplot(df_scaled, aes(x = Month, y = PC1)) +
  geom_boxplot(fill = "lightgreen") +
  labs(title = "Distribution of PC1 by Month", x = "Month", y = "PC1") +
  theme_minimal()

# Density Plot of PC1
ggplot(df_scaled, aes(x = PC1, fill = DayOfWeek)) +
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

