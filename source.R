required_packages <- c("dplyr", "ggplot2", "lubridate", "data.table",
                       "factoextra", "zoo", "Hmisc", "tidyr")

install_if_missing <- function(packages) {
  installed <- rownames(installed.packages())
  for (p in packages) {
    if (!(p %in% installed)) {
      install.packages(p, dependencies = TRUE)
    }
  }
}

install_if_missing(required_packages)

library(dplyr)
library(ggplot2)
library(lubridate)
library(data.table)
library(factoextra)
library(zoo)
library(Hmisc)
library(tidyr)

file_path <- "/Users/mahdi/Desktop/termproject/TermProjectData.txt"

df <- fread(file_path, header = TRUE, sep = ",", na.strings = "NA", stringsAsFactors = FALSE)
df <- as.data.frame(df)

# Data Inspection
cat("First 10 rows of the dataframe:\n")
print(head(df, 10))
cat("Column Names:\n")
print(colnames(df))

# Combining the Date and Time into DateTime
df$DateTime <- paste(df$Date, df$Time)
df$DateTime <- as.POSIXct(df$DateTime, format="%d/%m/%Y %H:%M:%S", tz = "UTC")

# Verifying Conversion
cat("DateTime conversion completed.\n")
str(df$DateTime)

# Filtering Data for Monday 09:00 AM to 12:00 PM
df <- df %>%
  filter(weekdays(DateTime) == "Monday" & 
           hour(DateTime) >= 9 & hour(DateTime) < 12)

cat("Extracted Time Window Data (09:00 AM to 12:00 PM on Monday):\n")
print(head(df))

# Converting Columns to Numeric
selected_numeric_cols <- c("Global_active_power", "Global_reactive_power", "Global_intensity", 
                           "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "Voltage")
df[selected_numeric_cols] <- lapply(df[selected_numeric_cols], as.numeric)

# Handling Missing Values
fill_na <- function(x) {
  x <- na.approx(x, na.rm = FALSE)
  x <- na.locf(x, na.rm = FALSE)
  x <- na.locf(x, na.rm = FALSE, fromLast = TRUE)
  return(x)
}
df[selected_numeric_cols] <- lapply(df[selected_numeric_cols], fill_na)

# Feature Engineering
df <- df %>%
  mutate(Global_submetering = Sub_metering_1 + Sub_metering_2 + Sub_metering_3)

# Updating Selected Numeric Columns for PCA
selected_numeric_cols <- c(selected_numeric_cols, "Global_submetering")

# Scaling and PCA
df_scaled <- scale(df[selected_numeric_cols])

pca_result <- prcomp(df_scaled, center = TRUE, scale. = TRUE)

# Visualization with PCA Components

# PCA biplot using factoextra
pca_plot <- fviz_pca_biplot(pca_result,
                            geom = "point",          # Shows data points
                            addEllipses = FALSE,     # No ellipses around groups
                            col.var = "red",         # Default color for all other variables
                            col.ind = "black",       # Color of points
                            repel = TRUE,            # Avoids text overlapping
                            arrow.size = 0.5,        # Size of the arrows
                            pointsize = 3,           # Size of the data points
                            labelsize = 5,           # Size of variable names
                            legend.title = "Variables") +
  theme(legend.position = "none")  

print(pca_plot)
