import pandas as pd
import numpy as np
import rdkit.Chem as Chem
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

df = pd.read_csv('mcule_purchasable_in_stock_prices_valid_smiles.csv')

plot = sns.displot(data=df[0:100000], x="price 1 (USD)", binwidth=15)#Remove [0:x] for full dataset
#plot.set(xlim=(0, 600), ylim=(0,2)) #If wanted plot with limits here
plt.show()
