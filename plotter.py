import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("output/omegas.txt", na_values="0.000000").dropna()
df.plot(kind="bar")
df.plot.box()
plt.show()