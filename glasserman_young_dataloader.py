import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import numpy as np

sns.set_theme()

data = pd.read_csv('data/glasserman_young_data.csv')

data.sort_values('w_i', ascending=False, inplace=True)
data['cum_w_i'] = data['w_i'].cumsum()

# Assets from other nodes in the network
internal_assets = data['Assets'] - data['c_i']

internal_assets.to_csv('data/glasserman_young_data_assets.csv')

# Set liabilities equal to assets
# Liabilities to other nodes in the network
internal_liabilities = internal_assets

# Calculate external liabilities
external_liabilities = internal_liabilities / data['beta_i'] - internal_liabilities

# Calculate External Assets
external_assets = data['c_i']

# Params are shape, loc, scale
wealth_params = scipy.stats.lognorm.fit(data['w_i'])
print('Shape: {}, Loc: {}, Scale: {}'.format(*wealth_params))

plt.figure()
sns.jointplot(external_assets, external_liabilities, kind='reg')

plt.figure()
sns.distplot(data['w_i'], kde=False, fit=scipy.stats.lognorm, label='Wealth')
plt.xlim(0, data['w_i'].max())
plt.legend()

plt.figure()
sns.distplot(external_assets, kde=False, fit=scipy.stats.powerlaw, label='External Assets')
sns.distplot(external_liabilities, kde=False, fit=scipy.stats.powerlaw, label='External Liabilities')
plt.xlim(0, external_assets.max())
plt.legend()

plt.figure()
sns.distplot(internal_assets, kde=False, fit=scipy.stats.lognorm, label='Internal Assets (= Internal Liabilities)')
plt.xlim(0, internal_assets.max())
plt.legend()

plt.figure()
plt.plot(np.linspace(0, 1, len(data)), data['cum_w_i'].to_numpy())
plt.ylabel('Cummulative Wealth')
plt.xlabel('Percentile')
plt.show()
