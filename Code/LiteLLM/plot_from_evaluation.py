import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("results_raw.csv")

categories = df.columns
x = np.arange(len(categories))
col_sum = []
for column in categories:
    col_sum.append(df[column].sum())

# Plotting segmented (stacked) bars
for label, row in df.iterrows():
    split_label = str(label).split(";")
    color = split_label[1] if len(split_label) > 1 else None

    col_sum -= row
    bottom = col_sum[:]
    plt.bar(x, row, bottom=bottom, label=split_label[0], color=color)

# Formatting
plt.xticks(x, categories, rotation=15, ha='right')
plt.yticks(np.arange(np.max(col_sum)))
plt.ylabel('Iterationen')
plt.title('Ergebnisse der variierten Modelle und Modellkonfigurationen')
plt.legend()

fig = plt.gcf()
plt.subplots_adjust(bottom=0.122)
plt.savefig("plot1.png", dpi=300)
plt.show()
