import pandas as pd
from sklearn.cluster import KMeans

data = pd.DataFrame({
    "age": [25, 35, 45, 23, 52],
    "income": [40000, 60000, 80000, 42000, 90000]
})

model = KMeans(n_clusters=2, random_state=42)
data["segment"] = model.fit_predict(data)

print(data)
