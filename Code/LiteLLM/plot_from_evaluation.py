import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("results.csv")

categories = df.columns
x = np.arange(len(categories))
col_sum = []
for column in categories:
    col_sum.append(df[column].sum())

# Plotting segmented (stacked) bars
for label, row in reversed([*df.iterrows()]):
    col_sum -= row
    bottom = col_sum[:]
    plt.bar(x, row, bottom=bottom, label=label)

# Formatting
plt.xticks(x, categories)
plt.ylabel('Value')
plt.title('Segmented Bar Chart')
plt.legend()

plt.show()
