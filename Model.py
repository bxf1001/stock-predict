import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet,Lasso,Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
data = pd.read_csv("big_tech_stock_prices.csv")

# Convert 'date' column to datetime type
data['date'] = pd.to_datetime(data['date'])

# Filter for Tesla data
tesla_data = data[data['stock_symbol'] == 'NVDA'].sort_values('date')

# --- RSI Calculation (Without talib) ---
def calculate_rsi_pandas(prices, period=14):
    """Calculates the Relative Strength Index (RSI) using pandas.

    Args:
        prices: A pandas Series of prices.
        period: The lookback period for the RSI calculation (default is 14).

    Returns:
        A pandas Series of RSI values.
    """

    # Calculate price changes
    deltas = prices.diff()

    # Separate gains and losses
    gains = deltas.where(deltas > 0, 0)
    losses = -deltas.where(deltas < 0, 0)

    # Calculate average gains and losses over the period
    avg_gain = gains.rolling(window=period).mean()
    avg_loss = losses.rolling(window=period).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

# --- MACD Calculation (Without talib) ---
def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    """Calculates the Moving Average Convergence Divergence (MACD) using pandas.

    Args:
        prices: A pandas Series of prices.
        fast_period: The period for the fast moving average (default is 12).
        slow_period: The period for the slow moving average (default is 26).
        signal_period: The period for the signal line (default is 9).

    Returns:
        A pandas Series of MACD values.
    """

    # Calculate the fast and slow moving averages
    fast_ma = prices.rolling(window=fast_period).mean()
    slow_ma = prices.rolling(window=slow_period).mean()

    # Calculate the MACD line
    macd_line = fast_ma - slow_ma

    # Calculate the signal line
    signal_line = macd_line.rolling(window=signal_period).mean()

    # Calculate the MACD histogram
    macd_histogram = macd_line - signal_line

    return macd_histogram

# --- Bollinger Bands Calculation (Without talib) ---
def calculate_bollinger_bands(prices, window=20):
    """Calculates Bollinger Bands using pandas.

    Args:
        prices: A pandas Series of prices.
        window: The lookback period for the moving average (default is 20).

    Returns:
        A tuple containing the upper band, middle band (moving average), and lower band.
    """

    # Calculate the moving average
    ma = prices.rolling(window=window).mean()

    # Calculate the standard deviation
    std = prices.rolling(window=window).std()

    # Calculate the upper and lower bands
    upper_band = ma + 2 * std
    lower_band = ma - 2 * std

    return upper_band, ma, lower_band

# --- Feature Engineering ---

tesla_data['close_shifted'] = tesla_data['close'].shift(50)
tesla_data['close_50_ma'] = tesla_data['close'].rolling(window=6).mean()
tesla_data['close_200_ma'] = tesla_data['close'].rolling(window=25).mean()
tesla_data['close_std_50'] = tesla_data['close'].rolling(window=6).std()
tesla_data['RSI'] = calculate_rsi_pandas(tesla_data['close'], period=14)
tesla_data['MACD'] = calculate_macd(tesla_data['close'], fast_period=6, slow_period=13, signal_period=5)
tesla_data['upper_band'], tesla_data['middle_band'], tesla_data['lower_band'] = calculate_bollinger_bands(tesla_data['close'], window=20)

tesla_data.dropna(inplace=True)

# Define features and target variable
X = tesla_data[['close_shifted', 'close_50_ma', 'close_200_ma', 'close_std_50', 'RSI', 'MACD', 'upper_band', 'middle_band', 'lower_band']]
y = tesla_data['close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Feature Selection (Using Random Forest) ---
rf = RandomForestRegressor(random_state=12)
rf.fit(X_train, y_train)
feature_importances = rf.feature_importances_

# Select features with importance above a threshold
important_features = X.columns[feature_importances > 0.05]  # Adjust threshold as needed

# --- Train ElasticNet Model with Selected Features ---
# Get the indices of the important features
important_feature_indices = [list(X.columns).index(feature) for feature in important_features]

# Select features using indices
X_train_selected = X_train[:, important_feature_indices]
X_test_selected = X_test[:, important_feature_indices]

# Define hyperparameters to tune
param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0,1000.0,10000.0,100000.0],
              'l1_ratio': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}

# Create grid search object
grid_search = GridSearchCV(ElasticNet(), param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)

# Perform hyperparameter tuning
grid_search.fit(X_train_selected, y_train)

# Get the best model and its hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Train the best model on the entire data
best_model.fit(X_train_selected, y_train)

# Make predictions
y_pred = best_model.predict(X_test_selected)

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

