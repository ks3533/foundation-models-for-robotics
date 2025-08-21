import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("results_raw.csv")

categories = df.columns
x = np.arange(len(categories))
col_sum = [df[column].sum() for column in categories]
remaining_col_sum = col_sum[:]
# Plotting segmented (stacked) bars
for label, row in df.iterrows():
    split_label = str(label).split(";")
    color = split_label[1] if len(split_label) > 1 else None

    remaining_col_sum -= row
    bottom = remaining_col_sum[:]
    plt.bar(x, row, bottom=bottom, label=split_label[0], color=color)

# Formatting
plt.xticks(x, categories, rotation=30, ha='right')
plt.yticks(np.arange(0, np.max(col_sum)+1, step=5))
plt.ylabel('Iterationen')
plt.title('Ergebnisse der variierten Modelle und Modellkonfigurationen')
plt.legend()

plt.subplots_adjust(bottom=0.45, left=0.15)
plt.savefig("plot1.png", dpi=300)
plt.show()
