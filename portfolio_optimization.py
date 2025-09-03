import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')

# -------------------------
# Section 1 - Setup
# -------------------------
tickers = ['COCHINSHIP.NS', 'RVNL.NS', 'POWERINDIA.NS', 'VEDL.NS', 'OIL.NS']
end_date = datetime.today()
start_date = end_date - timedelta(days=5 * 365)

# -------------------------
# Section 2 - Fetch Data
# -------------------------
adj_close_df = pd.DataFrame()

for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    adj_close_df[ticker] = data['Close']

# -------------------------
# Section 3 - Log Returns
# -------------------------
log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()

# -------------------------
# Section 4 - Covariance Matrix
# -------------------------
cov_matrix = log_returns.cov() * 252  # Annualized

# -------------------------
# Section 5 - Portfolio Metrics
# -------------------------
def standard_deviation(weights, cov_matrix):
    return np.sqrt(weights.T @ cov_matrix @ weights)

def expected_return(weights, log_returns):
    return np.sum(log_returns.mean() * weights) * 252

def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate=0.07):
    return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate=0.07):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

# -------------------------
# Section 6 - Optimization
# -------------------------
num_assets = len(tickers)
initial_weights = np.array([1/num_assets] * num_assets)
bounds = [(0, 0.4)] * num_assets
constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
risk_free_rate = 0.07

result = minimize(
    neg_sharpe_ratio,
    initial_weights,
    args=(log_returns, cov_matrix, risk_free_rate),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

optimal_weights = result.x
portfolio_return = expected_return(optimal_weights, log_returns)
portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
portfolio_sharpe = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)

# -------------------------
# Section 7 - Output Results
# -------------------------
print("\nOptimal Portfolio Weights:")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.4f}")
print(f"\nExpected Annual Return: {portfolio_return:.4f}")
print(f"Expected Volatility   : {portfolio_volatility:.4f}")
print(f"Sharpe Ratio           : {portfolio_sharpe:.4f}")

# -------------------------
# Section 8 - Plot Portfolio Weights
# -------------------------
plt.figure(figsize=(10, 6))
plt.bar(tickers, optimal_weights)
plt.title('Optimal Portfolio Weights')
plt.xlabel('Assets')
plt.ylabel('Weight')
plt.show()

# -------------------------
# Section 9 - Monte Carlo Simulation
# -------------------------
num_portfolios = 5000
results = np.zeros((3, num_portfolios))
weights_record = []

for i in range(num_portfolios):
    weights = np.random.dirichlet(np.ones(num_assets), size=1)[0]
    weights_record.append(weights)
    port_return = expected_return(weights, log_returns)
    port_volatility = standard_deviation(weights, cov_matrix)
    port_sharpe = sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

    results[0, i] = port_return
    results[1, i] = port_volatility
    results[2, i] = port_sharpe

# -------------------------
# Section 10 - Efficient Frontier Plot
# -------------------------
plt.figure(figsize=(12, 6))
plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', alpha=0.7)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(portfolio_volatility, portfolio_return, color='red', s=100, marker='*', label='Optimal Portfolio')
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier - Monte Carlo Simulation')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# Section 11 - Cumulative Returns
# -------------------------
cumulative_returns = (log_returns + 1).cumprod()

plt.figure(figsize=(12, 6))
for ticker in tickers:
    plt.plot(cumulative_returns[ticker], label=ticker)
plt.title('Cumulative Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# Section 12 - Correlation Heatmap
# -------------------------
plt.figure(figsize=(10, 6))
sns.heatmap(log_returns.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Asset Log Returns")
plt.show()
