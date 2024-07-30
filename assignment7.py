# %%
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# %%
data = pd.read_csv("../data/yields41.csv")
print(data.head())

# %%
corrMatrix = data.corr()
print(corrMatrix["yield"].sort_values())

# %%
features = data.columns[1:]
x = data.loc[:, features].values
x = StandardScaler().fit_transform(x)

# %%
pca = PCA()
principalComponents = pca.fit_transform(x)
explainedVariance = pca.explained_variance_ratio_

cumulativeVariance = np.cumsum(explainedVariance)
components = np.where(cumulativeVariance >= 0.8)[0][0] + 1

print(f"Components for 80%: {components}")

# %%
plt.figure(figsize=(10, 8))
plt.plot(
    range(1, len(explainedVariance) + 1), cumulativeVariance, marker="o", linestyle="--"
)
plt.title("Explained Variance")
plt.xlabel("Components")
plt.ylabel("Cumulative Variance")
plt.axvline(x=components, color="black")
plt.axhline(y=0.8, color="black")
plt.show()

# %%
loading = pd.DataFrame(pca.components_[0:2, :], columns=features).T
loading.columns = ["PC1", "PC2"]

# %%
plt.figure(figsize=(10, 7))
sns.scatterplot(data=loading, x="PC1", y="PC2", s=100)

for i in range(loading.shape[0]):
    plt.text(loading.PC1[i], loading.PC2[i], loading.index[i])

plt.title("Loadings Plot PC1 vs PC2")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.axhline(0, color="blue", linestyle="--")
plt.axvline(0, color="red", linestyle="--")
plt.grid(True)
plt.show()
