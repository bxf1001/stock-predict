import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv("big_tech_stock_prices.csv")

# Convert 'date' column to datetime type
data['date'] = pd.to_datetime(data['date'])
# Pivot the data to have dates as index and stock symbols as columns


# Prepare the data for a single stock (e.g., Apple)
apple_data = data[data['stock_symbol'] == 'AAPL']

# Feature engineering (using past 5 days' close prices as features)
apple_data['prev_close_1'] = apple_data['close'].shift(1)
apple_data['prev_close_2'] = apple_data['close'].shift(2)
apple_data['prev_close_3'] = apple_data['close'].shift(3)
apple_data['prev_close_4'] = apple_data['close'].shift(4)
apple_data['prev_close_5'] = apple_data['close'].shift(5)
apple_data['prev_close_6'] = apple_data['close'].shift(6)
apple_data['prev_close_7'] = apple_data['close'].shift(7)
apple_data['prev_close_8'] = apple_data['close'].shift(8)
apple_data['prev_close_9'] = apple_data['close'].shift(9)
apple_data['prev_close_10'] = apple_data['close'].shift(10)


# Drop rows with NaN values
apple_data.dropna(inplace=True)

# Define features and target variable
X = apple_data[['prev_close_1', 'prev_close_2', 'prev_close_3', 'prev_close_4', 'prev_close_5', 'prev_close_6', 'prev_close_7', 'prev_close_8', 'prev_close_9', 'prev_close_10']]
y = apple_data['close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameters to tune
param_grid = {'alpha': [0.1, 0.5, 1.0, 5.0]}

# Create grid search object
grid_search = GridSearchCV(Ridge(), param_grid, cv=5)

# Perform hyperparameter tuning
grid_search.fit(X_train, y_train)

# Get the best model and its hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Train the best model on the entire data
best_model.fit(X, y)

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Best Parameters: {best_params}')

# Plot the actual vs predicted prices
plt.figure(figsize=(14, 7))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title('Actual vs Predicted Closing Prices')
plt.show()