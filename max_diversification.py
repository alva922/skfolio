#https://github.com/skfolio/skfolio/blob/main/examples/4_maximum_diversification/plot_1_maximum_divesification.py

from plotly.io import show
from sklearn.model_selection import train_test_split

from skfolio import Population
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import EqualWeighted, MaximumDiversification
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()
X = prices_to_returns(prices)

X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)

model = MaximumDiversification()
model.fit(X_train)
model.weights_

bench = EqualWeighted()
bench.fit(X_train)
bench.weights_

ptf_model_train = model.predict(X_train)
ptf_bench_train = bench.predict(X_train)
print("Diversification Ratio:")
print(f"    Maximum Diversification model: {ptf_model_train.diversification:0.2f}")
print(f"    Equal Weighted model: {ptf_bench_train.diversification:0.2f}")

ptf_model_test = model.predict(X_test)
ptf_bench_test = bench.predict(X_test)

population = Population([ptf_model_test, ptf_bench_test])

fig = population.plot_composition()
show(fig)

population.plot_cumulative_returns()

population.summary()
