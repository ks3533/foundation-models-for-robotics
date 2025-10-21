import pandas as pd

# Read CSV
df = pd.read_csv("results_raw.csv", index_col=0)


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


# Decode categories
decoded_info = [decode_simulation(c) for c in df.columns]

# Build MultiIndex for columns: Model > Vision Type > Reasoning
columns_tuples = []
for d in decoded_info:
    model = d["model"]
    vision = d["vision_type"]
    if d["picture_every_tool_call"]:
        vision += "+tc"  # mark image+tool call
    reasoning = "+" if d["reasoning"] else "-"
    columns_tuples.append((model, vision, reasoning))

multi_index = pd.MultiIndex.from_tuples(columns_tuples, names=["Model", "Vision Type", "Reasoning"])
df.columns = multi_index

# Sort columns by Model > Vision Type > Reasoning (reasoning last)
df = df.reindex(sorted(df.columns, key=lambda x: (x[0], x[1], x[2])), axis=1)

# Transpose results so rows = Passes/Fails/Errors/Timeouts
result_values = df.T
result_values = result_values.T  # now rows = Passes/Fails/Errors/Timeouts

# Rename index to clean names
result_values.index = ["Passes", "Fails", "Errors", "Timeouts"]

# Export LaTeX with multicolumns
latex_code = result_values.to_latex(multicolumn=True,
                                    multicolumn_format='c',
                                    longtable=False,
                                    escape=False)
with open("comparison_table.tex", "w") as f:
    f.write(latex_code)
