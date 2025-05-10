#https://github.com/skfolio/skfolio/blob/main/examples/1_mean_risk/plot_13_factor_model.py

from plotly.io import show
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import train_test_split

from skfolio import Population, RiskMeasure
from skfolio.datasets import load_factors_dataset, load_sp500_dataset
from skfolio.moments import GerberCovariance, ShrunkMu
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EmpiricalPrior, FactorModel, LoadingMatrixRegression

prices = load_sp500_dataset()
factor_prices = load_factors_dataset()

X, y = prices_to_returns(prices, factor_prices)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model_factor_1 = MeanRisk(
    risk_measure=RiskMeasure.CVAR,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    prior_estimator=FactorModel(),
    portfolio_params=dict(name="Factor Model 1"),
)
model_factor_1.fit(X_train, y_train)
model_factor_1.weights_

model_factor_2 = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    prior_estimator=FactorModel(
        loading_matrix_estimator=LoadingMatrixRegression(
            linear_regressor=RidgeCV(fit_intercept=False), n_jobs=-1
            #linear_regressor=LassoCV(fit_intercept=False), n_jobs=-1
        )
    ),
    portfolio_params=dict(name="Factor Model 2"),
)
model_factor_2.fit(X_train, y_train)
model_factor_2.weights_

model_factor_3 = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    prior_estimator=FactorModel(
        factor_prior_estimator=EmpiricalPrior(
            mu_estimator=ShrunkMu(), covariance_estimator=GerberCovariance()
        )
    ),
    portfolio_params=dict(name="Factor Model 3"),
)
model_factor_3.fit(X_train, y_train)
model_factor_3.weights_


prior_estimator = model_factor_3.prior_estimator_


# We can access the prior model with:
prior_model = prior_estimator.prior_model_


# We can access the loading matrix with:
loading_matrix = prior_estimator.loading_matrix_estimator_.loading_matrix_


# Empirical Model
# ===============
# For comparison, we also create a Maximum Sharpe Ratio model using the default
# Empirical estimator:
model_empirical = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    portfolio_params=dict(name="Empirical"),
)
model_empirical.fit(X_train)
model_empirical.weights_



# ==========
# We predict all models on the test set:
ptf_factor_1_test = model_factor_1.predict(X_test)
ptf_factor_2_test = model_factor_2.predict(X_test)
ptf_factor_3_test = model_factor_3.predict(X_test)
ptf_empirical_test = model_empirical.predict(X_test)

population = Population(
    [ptf_factor_1_test, ptf_factor_2_test, ptf_factor_3_test, ptf_empirical_test]
)

fig = population.plot_cumulative_returns()
show(fig)


population.plot_composition()

population.summary()

plt.figure(figsize=(16, 6), dpi=80)
plt.rcParams.update({'font.size': 14})
plt.bar(cols,model_factor_1.weights_,label='Factor Model 1',color='r')        
plt.legend()
plt.grid()
plt.xlabel('Assets')
plt.ylabel('Portfolio Weights')
plt.show()

plt.figure(figsize=(16, 6), dpi=80)
plt.rcParams.update({'font.size': 14})
       
plt.bar(cols,model_factor_2.weights_,label='Factor Model 2',color='b')
plt.legend()
plt.grid()
plt.xlabel('Assets')
plt.ylabel('Portfolio Weights')
plt.show()

plt.figure(figsize=(16, 6), dpi=80)
plt.rcParams.update({'font.size': 14})

plt.bar(cols,model_factor_3.weights_,label='Factor Model 3',color='g')

plt.legend()
plt.grid()
plt.xlabel('Assets')
plt.ylabel('Portfolio Weights')
plt.show()

plt.figure(figsize=(16, 6), dpi=80)
plt.rcParams.update({'font.size': 14})

plt.bar(cols,model_empirical.weights_,label='Empirical',color='y')

plt.legend()
plt.grid()
plt.xlabel('Assets')
plt.ylabel('Portfolio Weights')
plt.show()
