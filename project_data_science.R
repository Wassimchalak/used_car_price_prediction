# ============================================================
# 1. LOAD AND INSPECT DATA
# ============================================================

# Read dataset
df <- read.csv("car_data.csv")

# Inspect original dataset
dim(df)
names(df)
str(df)
head(df)



# ============================================================
# 2. DATA PREPROCESSING
# ============================================================

library(tidyverse)

# Remove attributes not used in the analysis
df <- df %>%
  select(
    -Fuel.Type,
    -Location,
    -Insurance.Type
  )

# Check remaining attributes
names(df)
dim(df)

# Check missing values
colSums(is.na(df))

# Check duplicated rows
sum(duplicated(df))



# ============================================================
# 3. RANDOM SAMPLE OF 5,000 OBSERVATIONS
# ============================================================

set.seed(123)

df <- df[
  sample(
    seq_len(nrow(df)),
    5000
  ),
]

# Reset row numbers
rownames(df) <- NULL

dim(df)



# ============================================================
# 4. DATA TYPES
# ============================================================

# Convert qualitative variables to factors
df$Brand <- as.factor(df$Brand)
df$Model <- as.factor(df$Model)
df$Transmission.Type <- as.factor(df$Transmission.Type)
df$Color <- as.factor(df$Color)
df$Service.History <- as.factor(df$Service.History)
df$Previous.Accidents <- as.factor(df$Previous.Accidents)
df$Car.Type <- as.factor(df$Car.Type)

# Check structure
str(df)



# ============================================================
# 5. CREATE PRICE IN USD
# ============================================================

# Convert INR price to USD
df$Price.USD <- round(
  df$Price..INR. / 95.01
)

dim(df)



# ============================================================
# 6. SUMMARY OF QUANTITATIVE VARIABLES
# ============================================================

summary(df$Year)
summary(df$Odometer.Reading..km.)
summary(df$Number.of.Owners)
summary(df$Engine.Capacity..L.)
summary(df$Price.USD)



# ============================================================
# 7. EXPLORATORY DATA ANALYSIS
# ============================================================


# -------------------------
# Price distribution
# -------------------------

ggplot(df, aes(x = Price.USD)) +
  geom_histogram(
    binwidth = 5000,
    color = "black"
  ) +
  scale_x_continuous(
    breaks = seq(0, 105000, by = 5000)
  ) +
  labs(
    title = "Distribution of Car Prices",
    x = "Price (USD)",
    y = "Frequency"
  )


# -------------------------
# Price boxplot
# -------------------------

ggplot(df, aes(y = Price.USD)) +
  geom_boxplot() +
  scale_y_continuous(
    breaks = seq(0, 105000, by = 5000)
  ) +
  labs(
    title = "Boxplot of Car Prices",
    y = "Price (USD)"
  )


# -------------------------
# Year versus Price
# -------------------------

ggplot(
  df,
  aes(
    x = Year,
    y = Price.USD
  )
) +
  geom_point(alpha = 0.5) +
  scale_x_continuous(
    breaks = seq(2000, 2024, by = 2)
  ) +
  scale_y_continuous(
    breaks = seq(0, 105000, by = 5000)
  ) +
  labs(
    title = "Car Price versus Manufacturing Year",
    x = "Year",
    y = "Price (USD)"
  )


# -------------------------
# Year versus Price by Brand
# -------------------------

ggplot(
  df,
  aes(
    x = Year,
    y = Price.USD,
    color = Brand
  )
) +
  geom_point(alpha = 0.5) +
  scale_y_continuous(
    breaks = seq(0, 105000, by = 5000)
  ) +
  labs(
    title = "Car Price versus Manufacturing Year by Brand",
    x = "Year",
    y = "Price (USD)",
    color = "Brand"
  )


# -------------------------
# Odometer versus Price
# -------------------------

ggplot(
  df,
  aes(
    x = Odometer.Reading..km.,
    y = Price.USD,
    color = Year
  )
) +
  geom_point(alpha = 0.5) +
  scale_x_continuous(
    breaks = seq(
      0,
      max(
        df$Odometer.Reading..km.,
        na.rm = TRUE
      ),
      by = 20000
    )
  ) +
  scale_y_continuous(
    breaks = seq(0, 105000, by = 5000)
  ) +
  labs(
    title = "Car Price versus Odometer Reading by Year",
    x = "Odometer Reading (km)",
    y = "Price (USD)",
    color = "Year"
  )



# -------------------------
# Categorical variables
# -------------------------

categorical_data <- df %>%
  select(
    Price.USD,
    Brand,
    Transmission.Type,
    Color,
    Service.History,
    Previous.Accidents,
    Car.Type
  ) %>%
  pivot_longer(
    cols = -Price.USD,
    names_to = "Attribute",
    values_to = "Category"
  )

ggplot(
  categorical_data,
  aes(
    x = Category,
    y = Price.USD
  )
) +
  geom_boxplot() +
  facet_wrap(
    ~ Attribute,
    scales = "free_x",
    ncol = 2
  ) +
  scale_y_continuous(
    breaks = seq(0, 105000, by = 5000)
  ) +
  labs(
    title = "Car Price by Categorical Attributes",
    x = "",
    y = "Price (USD)"
  ) +
  theme(
    axis.text.x = element_text(
      angle = 45,
      hjust = 1
    )
  )



# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

set.seed(456)

train_index <- sample(
  seq_len(nrow(df)),
  size = 0.8 * nrow(df)
)

train <- df[train_index, ]
test <- df[-train_index, ]

dim(train)
dim(test)



# Avoid scientific notation
options(scipen = 999)



# ============================================================
# 9. SIMPLE LINEAR REGRESSION
# ============================================================
#
# Three separate SLR models are tested:
# 1. Engine Capacity
# 2. Manufacturing Year
# 3. Odometer Reading
#
# Each model contains only ONE predictor.
# ============================================================


# ------------------------------------------------------------
# SLR 1: Engine Capacity
# ------------------------------------------------------------

slr_engine <- lm(
  Price.USD ~ Engine.Capacity..L.,
  data = train
)

summary(slr_engine)

# R-squared
summary(slr_engine)$r.squared

# Residual Standard Error
summary(slr_engine)$sigma


# Test predictions
slr_engine_predictions <- predict(
  slr_engine,
  newdata = test
)

# Test MSE
slr_engine_mse <- mean(
  (test$Price.USD - slr_engine_predictions)^2
)

# Test RMSE
slr_engine_rmse <- sqrt(
  slr_engine_mse
)

slr_engine_mse
slr_engine_rmse



# ------------------------------------------------------------
# SLR 2: Manufacturing Year
# ------------------------------------------------------------

slr_year <- lm(
  Price.USD ~ Year,
  data = train
)

summary(slr_year)

# R-squared
summary(slr_year)$r.squared

# Residual Standard Error
summary(slr_year)$sigma


# Test predictions
slr_year_predictions <- predict(
  slr_year,
  newdata = test
)

# Test MSE
slr_year_mse <- mean(
  (test$Price.USD - slr_year_predictions)^2
)

# Test RMSE
slr_year_rmse <- sqrt(
  slr_year_mse
)

slr_year_mse
slr_year_rmse



# ------------------------------------------------------------
# SLR 3: Odometer Reading
# ------------------------------------------------------------

slr_odometer <- lm(
  Price.USD ~ Odometer.Reading..km.,
  data = train
)

summary(slr_odometer)

# R-squared
summary(slr_odometer)$r.squared

# Residual Standard Error
summary(slr_odometer)$sigma


# Test predictions
slr_odometer_predictions <- predict(
  slr_odometer,
  newdata = test
)

# Test MSE
slr_odometer_mse <- mean(
  (test$Price.USD - slr_odometer_predictions)^2
)

# Test RMSE
slr_odometer_rmse <- sqrt(
  slr_odometer_mse
)

slr_odometer_mse
slr_odometer_rmse



# ------------------------------------------------------------
# Compare the three SLR models
# ------------------------------------------------------------

slr_comparison <- data.frame(
  Model = c(
    "Engine Capacity",
    "Year",
    "Odometer Reading"
  ),
  
  R_Squared = c(
    summary(slr_engine)$r.squared,
    summary(slr_year)$r.squared,
    summary(slr_odometer)$r.squared
  ),
  
  MSE = c(
    slr_engine_mse,
    slr_year_mse,
    slr_odometer_mse
  ),
  
  RMSE = c(
    slr_engine_rmse,
    slr_year_rmse,
    slr_odometer_rmse
  )
)
# ------------------------------------------------------------
# Select the best SLR model
# ------------------------------------------------------------

# Year produced the highest R-squared
# and the lowest test RMSE among the three SLR models.

slr_model <- slr_year

slr_predictions <- slr_year_predictions

slr_mse <- slr_year_mse

slr_rmse <- slr_year_rmse

slr_comparison
# ============================================================
# 10. MANUAL MULTIPLE LINEAR REGRESSION
# ============================================================

mlr_model <- lm(
  Price.USD ~
    Year +
    Odometer.Reading..km. +
    Brand,
  data = train
)

summary(mlr_model)


# Test-set predictions
mlr_predictions <- predict(
  mlr_model,
  newdata = test
)

# Test MSE
mlr_mse <- mean(
  (test$Price.USD - mlr_predictions)^2
)

mlr_mse

# Test RMSE
mlr_rmse <- sqrt(mlr_mse)

mlr_rmse



# ============================================================
# 11. AUTOMATIC MLR VARIABLE SELECTION
# ============================================================

# Intercept-only model
null_model <- lm(
  Price.USD ~ 1,
  data = train
)


# Largest candidate MLR
full_model <- lm(
  Price.USD ~
    Year +
    Odometer.Reading..km. +
    Brand +
    Transmission.Type +
    Color +
    Number.of.Owners +
    Service.History +
    Previous.Accidents +
    Engine.Capacity..L. +
    Car.Type,
  data = train
)


# Stepwise AIC selection
best_mlr <- step(
  null_model,
  scope = list(
    lower = formula(null_model),
    upper = formula(full_model)
  ),
  direction = "both"
)

formula(best_mlr)
summary(best_mlr)


# Test predictions
best_mlr_predictions <- predict(
  best_mlr,
  newdata = test
)

# Test MSE
best_mlr_mse <- mean(
  (test$Price.USD - best_mlr_predictions)^2
)

best_mlr_mse

# Test RMSE
best_mlr_rmse <- sqrt(best_mlr_mse)

best_mlr_rmse


# ============================================================
# 12. CLEAN INTERACTION MODEL
# ============================================================

# Consider all two-way interactions except Brand × Car.Type,
# which produced exact linear dependencies

interaction_full_clean <- update(
  best_mlr,
  . ~ .^2 - Brand:Car.Type
)

# Stepwise selection of useful interactions
best_interaction <- step(
  best_mlr,
  scope = list(
    lower = formula(best_mlr),
    upper = formula(interaction_full_clean)
  ),
  direction = "both"
)

# Final selected formula
formula(best_interaction)

# Check for aliased coefficients
sum(
  is.na(
    coef(best_interaction)
  )
)
#performance:
interaction_predictions <- predict(
  best_interaction,
  newdata = test
)

interaction_mse <- mean(
  (test$Price.USD - interaction_predictions)^2
)

interaction_rmse <- sqrt(
  interaction_mse
)

interaction_mse
interaction_rmse
# ============================================================
# 13. INITIAL POLYNOMIAL MODEL
# ============================================================

poly_model <- update(
  best_interaction,
  . ~ . + I(Year^2)
)

summary(poly_model)

# Check that no coefficients are aliased
sum(
  is.na(
    coef(poly_model)
  )
)
# Polynomial model predictions

poly_predictions <- predict(
  poly_model,
  newdata = test
)

# Test MSE
poly_mse <- mean(
  (test$Price.USD - poly_predictions)^2
)

# Test RMSE
poly_rmse <- sqrt(
  poly_mse
)

poly_mse
poly_rmse
# ============================================================
# 15. MODEL COMPARISON
# ============================================================

model_comparison <- data.frame(
  Model = c(
    "SLR",
    "MLR",
    "Interaction",
    "Polynomial"
  ),
  
  MSE = c(
    slr_mse,
    best_mlr_mse,
    interaction_mse,
    poly_mse
  ),
  
  RMSE = c(
    slr_rmse,
    best_mlr_rmse,
    interaction_rmse,
    poly_rmse
  )
)

model_comparison



# ============================================================
# 16. REGRESSION DIAGNOSTICS
# ============================================================


# ------------------------------------------------------------
# Diagnostic 1:
# Residuals versus Fitted
#
# Used to investigate:
# - Non-linearity
# - Heteroscedasticity
# ------------------------------------------------------------

par(
  mfrow = c(2, 2)
)

plot(
  slr_model,
  which = 1,
  main = "SLR: Residuals vs Fitted"
)

plot(
  best_mlr,
  which = 1,
  main = "MLR: Residuals vs Fitted"
)

plot(
  best_interaction,
  which = 1,
  main = "Interaction: Residuals vs Fitted"
)

plot(
  poly_model,
  which = 1,
  main = "Polynomial: Residuals vs Fitted"
)

par(
  mfrow = c(1, 1)
)





# ------------------------------------------------------------
# Diagnostic 2:
# Normal Q-Q plots
#
# Used to investigate:
# - Unusual residuals
# - Outliers
# - Normality of residuals
# ------------------------------------------------------------

par(
  mfrow = c(2, 2)
)

plot(
  slr_model,
  which = 2,
  main = "SLR: Normal Q-Q"
)

plot(
  best_mlr,
  which = 2,
  main = "MLR: Normal Q-Q"
)

plot(
  best_interaction,
  which = 2,
  main = "Interaction: Normal Q-Q"
)

plot(
  poly_model,
  which = 2,
  main = "Polynomial: Normal Q-Q"
)

par(
  mfrow = c(1, 1)
)





# ------------------------------------------------------------
# Diagnostic 3:
# Standardized residuals
#
# Used to identify potential outliers
# A common rule is |standardized residual| > 3
# ------------------------------------------------------------

# Find potential outliers in each model
slr_outliers <- which(
  abs(rstandard(slr_model)) > 3
)

mlr_outliers <- which(
  abs(rstandard(best_mlr)) > 3
)

interaction_outliers <- which(
  abs(rstandard(best_interaction)) > 3
)

poly_outliers <- which(
  abs(rstandard(poly_model)) > 3
)


# Number of potential outliers in each model
c(
  SLR = length(slr_outliers),
  MLR = length(mlr_outliers),
  Interaction = length(interaction_outliers),
  Polynomial = length(poly_outliers)
)


# Show observations identified as potential outliers
# in the final polynomial model

head(
  poly_outliers,
  20
)




# Show the corresponding standardized residuals

head(
  sort(
    abs(rstandard(poly_model)),
    decreasing = TRUE
  ),
  10
)




# Show the 10 most extreme standardized residuals
# from the final polynomial model

poly_standardized <- rstandard(poly_model)

top_outliers <- data.frame(
  Observation = as.integer(names(
    sort(
      abs(poly_standardized),
      decreasing = TRUE
    )[1:10]
  )),
  Standardized.Residual = poly_standardized[
    names(
      sort(
        abs(poly_standardized),
        decreasing = TRUE
      )[1:10]
    )
  ]
)

top_outliers





# ------------------------------------------------------------
# Diagnostic 4:
# Leverage and Cook's Distance
#
# Used to identify influential observations
# ------------------------------------------------------------

# Cook's distance for the final polynomial model
poly_cooks <- cooks.distance(poly_model)

# Common cutoff: 4 / n
cook_cutoff <- 4 / nrow(train)

cook_cutoff

# Find influential observations
influential_points <- which(
  poly_cooks > cook_cutoff
)

# Number of influential observations
length(influential_points)


# Show the 10 largest Cook's distances

head(
  sort(
    poly_cooks,
    decreasing = TRUE
  ),
  10
)



# ------------------------------------------------------------
# Diagnostic 5:
# Leverage
#
# Used to identify observations with unusual predictor values
# ------------------------------------------------------------

# Calculate leverage values
poly_leverage <- hatvalues(poly_model)

# Effective number of estimated parameters
p <- poly_model$rank

# Number of observations used in the model
n <- nobs(poly_model)

# Common leverage cutoff: 2p / n
leverage_cutoff <- 2 * p / n

leverage_cutoff

# Find high-leverage observations
high_leverage <- which(
  poly_leverage > leverage_cutoff
)

# Number of high-leverage observations
length(high_leverage)




# Show the 10 observations with the highest leverage

head(
  sort(
    poly_leverage,
    decreasing = TRUE
  ),
  10
)




# ------------------------------------------------------------
# Diagnostic 6:
# Multicollinearity
#
# VIF / GVIF for predictors in the selected MLR model

install.packages("car")

library(car)

vif_results <- vif(best_mlr)

vif_results




# ------------------------------------------------------------
# Diagnostic summary across all four models
# ------------------------------------------------------------

models <- list(
  SLR = slr_model,
  MLR = best_mlr,
  Interaction = best_interaction,
  Polynomial = poly_model
)

diagnostic_summary <- data.frame(
  Model = names(models),
  
  Outliers = sapply(
    models,
    function(m) sum(abs(rstandard(m)) > 3)
  ),
  
  High_Leverage = sapply(
    models,
    function(m) {
      cutoff <- 2 * m$rank / nobs(m)
      sum(hatvalues(m) > cutoff)
    }
  ),
  
  Influential_Cooks = sapply(
    models,
    function(m) {
      cutoff <- 4 / nobs(m)
      sum(cooks.distance(m) > cutoff)
    }
  )
)

diagnostic_summary





# ------------------------------------------------------------
# Diagnostic 7:
# Scale-Location plots
#
# Used to assess heteroscedasticity
# A roughly horizontal red line with even spread is desirable
# ------------------------------------------------------------

par(mfrow = c(2, 2))

plot(
  slr_model,
  which = 3,
  main = "SLR: Scale-Location"
)

plot(
  best_mlr,
  which = 3,
  main = "MLR: Scale-Location"
)

plot(
  best_interaction,
  which = 3,
  main = "Interaction: Scale-Location"
)

plot(
  poly_model,
  which = 3,
  main = "Polynomial: Scale-Location"
)

par(mfrow = c(1, 1))






# ------------------------------------------------------------
# Diagnostic 8:
# Aliased coefficients / exact multicollinearity
#
# Check whether any coefficients cannot be uniquely estimated
# ------------------------------------------------------------

c(
  Interaction_Aliased =
    sum(is.na(coef(best_interaction))),
  
  Polynomial_Aliased =
    sum(is.na(coef(poly_model)))
)



# Show coefficients that could not be uniquely estimated
# in the interaction model

names(
  coef(best_interaction)[
    is.na(coef(best_interaction))
  ]
)