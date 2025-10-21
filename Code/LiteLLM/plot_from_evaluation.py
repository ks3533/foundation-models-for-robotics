from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def decode_simulation(name: str) -> dict:
    """Decode a simulation name into structured parts and human-readable text."""

    # Flags
    is_reasoning = name.endswith("_reasoning")
    has_picture_call = "picture_every_tool_call" in name

    # Remove reasoning / picture tokens to isolate base
    base = name.replace("_reasoning", "")
    base = base.replace("_picture_every_tool_call", "")

    # Vision type
    if base.startswith("json_vision"):
        vision_type = "JSON scene desc."
        rest = base.replace("json_vision_", "")
    elif base.startswith("no_vision"):
        vision_type = "no vision"
        rest = base.replace("no_vision_", "")
    elif base.startswith("vision_legacy"):
        vision_type = "image"
        rest = base.replace("vision_legacy_", "")
    elif base.startswith("vision"):
        vision_type = "scene desc."
        rest = base.replace("vision_", "")
    else:
        vision_type = "unknown"
        rest = base

    # Model name cleanup
    model_name = rest.replace("_", "")

    # Human-readable parts
    description = vision_type
    if has_picture_call:
        description += " every tc"
    description += ", " + model_name
    if is_reasoning:
        description += " + r"

    def description_creator(omit_reasoning=False, omit_model=False):
        description_ = vision_type
        if has_picture_call:
            description_ += " + image every tool call"
        description_ += ", " + model_name if not omit_model else ""
        if is_reasoning and not omit_reasoning:
            description_ += " + reasoning"
        return description_

    return {
        "raw": name,
        "vision_type": vision_type,
        "model": model_name,
        "reasoning": is_reasoning,
        "picture_every_tool_call": has_picture_call,
        "human_readable": description,
        "description": description_creator
    }


df = pd.read_csv("results_raw.csv")

# first sort the variations by every variable that can change, TODO last by those who are split by
df.sort_index(
    axis=1,
    key=lambda cols: cols.map(lambda c: itemgetter("model", "vision_type", "reasoning")(decode_simulation(c))),
    inplace=True
)

categories = df.columns
decoded_categories = [decode_simulation(category) for category in categories]

# split_by = {"reasoning"}
split_by = []
num_subplots = 1
for split_category in split_by:
    unique_values = set()
    for category in decoded_categories:
        unique_values.add(category[split_category])
    num_subplots *= len(unique_values)

# try finding entries with only categories from split_by mismatched (the method relies on the sorting TODO above)
for i, decoded in enumerate(decoded_categories):
    # if i != len(decoded_categories)-1:
    # if all([decoded[cat] == decoded_categories[i+1][cat] for
    # cat in {"vision_type", "model", "reasoning", "picture_every_tool_call"} - split_by]):

    pass

fig, axes = plt.subplots(1, num_subplots, figsize=(8*num_subplots, 5))


x = np.arange(len(categories))
col_sum = [df[column["raw"]].sum() for column in decoded_categories]
remaining_col_sum = col_sum[:]

decoded_names = [category["human_readable"] for category in decoded_categories]
# Plotting segmented (stacked) bars
for label, row in df.iterrows():
    split_label = str(label).split(";")
    color = split_label[1] if len(split_label) > 1 else None

    remaining_col_sum -= row
    bottom = remaining_col_sum[:]
    plt.bar(x, row, bottom=bottom, label=split_label[0], color=color)

# Formatting
plt.xticks(x, decoded_names, rotation=45, ha='right')
plt.yticks(np.arange(0, np.max(col_sum)+1, step=5))
plt.ylabel('Iterationen')
plt.title('Ergebnisse der variierten Modelle und Modellkonfigurationen')
plt.legend(loc="lower left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()

# plt.subplots_adjust(bottom=0.55, left=0.17)
plt.savefig("plot4.png", dpi=300)
plt.show()
