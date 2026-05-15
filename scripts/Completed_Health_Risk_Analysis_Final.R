
# =============================================================================================
# HEALTHCARE RISK PREDICTION AND STATISTICAL MODELLING INSPIRED BY MY PREVIOUS PROJECTS IN NHS
# =============================================================================================
# Author  : Fungai Mupezeni
# Purpose : Healthcare data analysis, statistical modelling,
#           machine learning, PCA, and Data visualisation
# Language: R
#=====================================================

# Global chunk options
knitr::opts_chunk$set(
  echo = TRUE
)

# Set project root directory
knitr::opts_knit$set(
  root.dir = ".."
)

# =====================================================
required_packages <- c(
  "tidyverse",
  "caret",
  "randomForest",
  "FactoMineR",
  "factoextra",
  "ggplot2",
  "reshape2",
  "GGally",
  "rmarkdown",
  "MiKTeX",
  "TinyTeX",
  "MacTeX"
)

installed_packages <- rownames(installed.packages())

for (package in required_packages) {
  if (!(package %in% installed_packages)) {
    install.packages(
      package,
      repos = "https://cloud.r-project.org"
    )
  }
}

# =====================================================
# INSTALL REQUIRED PACKAGES (THAT YOU CAN RUN ONCE ONLY)
# =====================================================

#=================================================================
# I START BY LOADING LIBRARIES, THAT POWERS DIFFERENT FUNCTIONS
#==================================================================

library(tidyverse)
library(caret)
library(randomForest)
library(FactoMineR)
library(factoextra)
library(ggplot2)
library(reshape2)
library(GGally)
library(rmarkdown)

# ==================================================================
# LOAD DATASET (Is the file that contain data in different formats)
# ==================================================================

healthcare_data <- read.csv(
  "data/processed/healthcare_data.csv"
)

# This help display current working directory
getwd()
list.files(recursive = TRUE)
# Preview dataset see few roles and the data contained
head(healthcare_data)

# Checking the Dataset structure
str(healthcare_data)


# =====================================================
# DATA CLEANING/WRANGLING AS PART OF AND PREPROCESSING
# =====================================================

# Remove duplicate observations from the entire dataset
healthcare_data <- distinct(
  healthcare_data
)

# Checking and Removing missing values from the dataset
healthcare_data <- na.omit(
  healthcare_data
)

# Summary statistics to check the data distribution using descriptive statistics 
summary(healthcare_data)

# ==================================================================================
# TRAIN / TEST SPLIT (I used this to divide a dataset into two subsets) 80:20% ratio
# ==================================================================================

#================================================================================================
#I have used set seed ensures that the same random numbers are generated every time the code runs
set.seed(123)
#================================================================================================

#================================================================================================
#This code creates indices for splitting the dataset into training data (80%) and testing data (20%)
train_index <- createDataPartition(
  healthcare_data$risk_score,
  p    = 0.80,
  list = FALSE
)

training_data <- healthcare_data[
  train_index,
]

testing_data <- healthcare_data[
  -train_index,
]
#===========================================================================================

#===========================================================================================
# HERE I HAVE INTRODUCED RANDOM FOREST MACHINE LEARNING MODEL 
#The code below trains a Random Forest classification model using the training dataset .
#My goal here is to predict healthcare risk outcomes (risk_score) based on other variables
# ===========================================================================================

random_forest_model <- randomForest(
  as.factor(risk_score) ~ .,
  data = training_data
)

# Generate predictions
predictions <- predict(
  random_forest_model,
  testing_data
)
#=======================================================================================
# In order to evaluate this model, I employed the use of Confusion Matrix
#The reason being to check the performance of the machine learning classification model(RF)
#==========================================================================================
confusion_matrix <- confusionMatrix(
  predictions,
  as.factor(testing_data$risk_score)
)
#Here I printed the results
print(confusion_matrix)

# Then I Saved model
saveRDS(
  random_forest_model,
  file = "outputs/models/random_forest_model.rds"
)

# ============================================================================================
# PRINCIPAL COMPONENT ANALYSIS (PCA)
# I often use PCA for dimensionality reduction technique , feature selection data visualisation 
# Learned during my Data Science Internship at the National Center for Social research (NATCEN)
# =============================================================================================

# Select numerical variables and left out the target variable risk_core as a factor(Categorical)
pca_data <- healthcare_data[, c(
  "age",
  "bmi",
  "blood_pressure",
  "cholesterol",
  "glucose",
  "heart_rate",
  "smoking_score",
  "physical_activity"
)]

# Here I ran Principal Component Analysis (PCA)
pca_results <- PCA(
  pca_data,
  scale.unit = TRUE,
  graph      = FALSE
)

# Getting the PCA summary
summary(pca_results)


# ========================================================================
# HERE I CHOOSE TO DEMONSTRATE THE LINEAR REGRESSION MODEL BUILDING
# I used this to examine the relationship between healthcare risk outcomes
# ========================================================================

linear_model <- lm(
  risk_score ~ age + bmi + blood_pressure,
  data = healthcare_data
)

# To obtain detailed statistical summary of the linear regression model, I used summary function
summary(linear_model)
#==============================================================================================


# ==============================================================================
# LOGISTIC REGRESSION MODEL (GLM)
# ==============================================================================

logistic_model <- glm(
  risk_score ~ age + bmi + blood_pressure,
  family = binomial,
  data   = healthcare_data
)

summary(logistic_model)


# ==============================================================================
# DATA VISUALIZATION, I HAVE DECIDED TO USE DIFFERENCE TYPES OF REPRESENTATION
# ==============================================================================

# -----------------------------------------------------
#  TO SHOW BMI DISTRIBUTION USING A HISTOGRAM
# -----------------------------------------------------

bmi_plot <- ggplot(
  healthcare_data,
  aes(x = bmi)
) +
  geom_histogram(
    bins  = 30,
    fill  = "brown",
    color = "black",
    alpha = 0.8
  ) +
  labs(
    title = "BMI Distribution",
    x     = "BMI",
    y     = "Frequency"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(
      hjust = 0.5,
      face  = "bold",
      size  = 14
    ),
    axis.title = element_text(
      face = "bold"
    )
  )

# Display plot
print(bmi_plot)
#===============================================================================
# After plotting I Saved the figure in file name given below
ggsave(
  filename = "outputs/figures/bmi_distribution.png",
  plot     = bmi_plot,
  width    = 8,
  height   = 6,
  dpi      = 300
)
#===============================================================================

# ------------------------------------------------------------------------------
# I EMPLOYED THE USE OF A SCATTERPLOT TO SHOW AGE VS CHOLESTEROL 
# ------------------------------------------------------------------------------

scatter_plot <- ggplot(
  healthcare_data,
  aes(
    x     = age,
    y     = cholesterol,
    color = age
  )
) +
  geom_point(
    size  = 3,
    alpha = 0.7
  ) +
  scale_color_gradient(
    low  = "blue",
    high = "red"
  ) +
  labs(
    title = "Age vs Cholesterol",
    x     = "Age",
    y     = "Cholesterol Level"
  ) +
  theme_minimal()

print(scatter_plot)
#=================================================================================
# I Saved the figure in file name given below
ggsave(
  filename = "outputs/figures/age_vs_cholesterol.png",
  plot     = scatter_plot,
  width    = 8,
  height   = 6,
  dpi      = 300
)
#================================================================================

# -----------------------------------------------------
# I DECIDED TO SHOW VARIABLE CORRELATION ON HEATMAP
# -----------------------------------------------------

# Here I generated a correlation matrix
correlation_matrix <- cor(
  healthcare_data
)

# Then I had to Convert matrix to long format by melting the Matrix
melted_correlation <- melt(
  correlation_matrix
)

# I then Created a heatmap
correlation_heatmap <- ggplot(
  melted_correlation,
  aes(
    x    = Var1,
    y    = Var2,
    fill = value
  )
) +
  geom_tile() +
  scale_fill_gradient2(
    low      = "blue",
    mid      = "white",
    high     = "red",
    midpoint = 0
  ) +
  labs(
    title = "Correlation Heatmap"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(
      angle = 45,
      hjust = 1
    )
  )
#This printout the heatmap
print(correlation_heatmap)
#===============================================================================
# I Saved the heat map in the outputs ,figures folder
ggsave(
  filename = "outputs/figures/correlation_heatmap.png",
  plot     = correlation_heatmap,
  width    = 8,
  height   = 6,
  dpi      = 300
)
#===============================================================================

# -----------------------------------------------------
# I SHOWED BMI IN A BOXPLOT BY RISK SCORE
# -----------------------------------------------------

bmi_boxplot <- ggplot(
  healthcare_data,
  aes(
    x    = as.factor(risk_score),
    y    = bmi,
    fill = as.factor(risk_score)
  )
) +
  geom_boxplot() +
  scale_fill_manual(
    values = c("skyblue", "tomato")
  ) +
  labs(
    title = "BMI Distribution by Risk Score",
    x     = "Risk Score",
    y     = "BMI"
  ) +
  theme_minimal()

print(bmi_boxplot)

# I save this in the same folder as other outputs
ggsave(
  filename = "outputs/figures/bmi_boxplot.png",
  plot     = bmi_boxplot,
  width    = 8,
  height   = 6,
  dpi      = 300
)

# -----------------------------------------------------
# HERE I CHECKED THE BMI  BY RISK GROUP ON A HISTOGRAM
# -----------------------------------------------------

risk_histogram <- ggplot(
  healthcare_data,
  aes(
    x    = bmi,
    fill = as.factor(risk_score)
  )
) +
  geom_histogram(
    bins     = 30,
    alpha    = 0.7,
    position = "identity"
  ) +
  scale_fill_manual(
    values = c("cyan", "magenta")
  ) +
  labs(
    title = "BMI Histogram by Risk Group",
    x     = "BMI",
    y     = "Frequency"
  ) +
  theme_minimal()

print(risk_histogram)
#------------------------------------------------------------
# I saved this in the same folder as other outputs
ggsave(
  filename = "outputs/figures/risk_histogram.png",
  plot     = risk_histogram,
  width    = 8,
  height   = 6,
  dpi      = 300
)


# =====================================================
# I USED PCA FOR DATA VISUALIZATIONS
# =====================================================

# I first Converted the target variable to factor which is the risk_score
healthcare_data$risk_score <- as.factor(
  healthcare_data$risk_score
)

# I presented this in a Scree Plot visualisation to capture the variance
scree_plot <- fviz_eig(
  pca_results,
  addlabels = TRUE,
  barfill   = "steelblue",
  barcolor  = "black"
) +
  ggtitle(
    "Scree Plot - Explained Variance"
  )

print(scree_plot)

#I Saved the scree plot in the link below with other figures
ggsave(
  filename = "outputs/figures/scree_plot.png",
  plot     = scree_plot,
  width    = 8,
  height   = 6,
  dpi      = 300
)

# -----------------------------------------------------
# PCA INDIVIDUALS PLOT
# -----------------------------------------------------

pca_individual_plot <- fviz_pca_ind(
  pca_results,
  geom.ind    = "point",
  habillage   = healthcare_data$risk_score,
  palette     = c("blue", "red"),
  addEllipses = TRUE
)

print(pca_individual_plot)

# Saved PCA individuals plot in the outputs/figures folder
ggsave(
  filename = "outputs/figures/pca_individual_plot.png",
  plot     = pca_individual_plot,
  width    = 8,
  height   = 6,
  dpi      = 300
)

# =====================================================
# LASTLY I SAVED ALL OTHER REPORTS OUTPUTS IN THE REPORTS FOLDER PATH
#Regression summary,Confusion matrix,PCA eigenvalues
# =====================================================

# I Saved regression summary
sink(
  "outputs/reports/linear_regression_summary.txt"
)

summary(linear_model)

sink()

# I Saved confusion matrix
capture.output(
  confusion_matrix,
  file = "outputs/reports/confusion_matrix.txt"
)

# Save PCA eigenvalues
write.csv(
  pca_results$eig,
  "outputs/reports/pca_eigenvalues.csv"
)

# =====================================================
# SESSION INFORMATION
# =====================================================
R.version.string
sessionInfo()