#Naive Inverse-Volatility Allocation
#Basic Imports

from sklearn.model_selection import train_test_split

from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import InverseVolatility
from skfolio.preprocessing import prices_to_returns

#Data preparation

prices = load_sp500_dataset()

X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)

# Model Training

model = InverseVolatility()
model.fit(X_train)

# Model Test Predictions 

portfolio = model.predict(X_test)

# Benchmark Training

benchmark = EqualWeighted(portfolio_params=dict(name="Equal Weighted"))
benchmark.fit(X_train)

#Benchmark Test Predictions

pred_bench = benchmark.predict(X_test)

print(portfolio.cvar)
print(pred_bench.cvar)


print(portfolio.annualized_sharpe_ratio)
print(pred_bench.annualized_sharpe_ratio)

population = Population([portfolio, pred_bench])
population.plot_composition()

fig = population.plot_cumulative_returns()
show(fig)

population.summary()
