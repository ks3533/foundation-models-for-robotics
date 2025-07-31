from pathlib import Path

batches = []

passes_list = []
fails_list = []
errors_list = []
timeout_list = []

ai_fail = []
sim_fail = []

for path in Path("Logs").iterdir():
    if "_" in path.name:
        batches.append(path.name)
        fails = 0
        passes = 0
        errors = 0
        total = 0
        for iteration in path.iterdir():
            total += 1
            with open(iteration/"RobocasaLLM.log") as f:
                log = f.read()
            if log[-2] == "R":  # Error
                errors += 1
            elif log[-2] == "S":  # Success
                passes += 1
            elif log[-2] == "L":  # Fail
                fails += 1
            else:  # Timeout
                pass

        passes_list.append(str(passes))
        fails_list.append(str(fails))
        errors_list.append(str(errors))
        timeout_list.append(str(total-passes-errors-fails))
        print(f"{path.name}: {passes} passes, {fails} fails, {errors} errors, {total-passes-errors-fails} "
              f"timeouts/unknown of {total} total")
with open("results_raw.csv", "w") as f:
    f.write(",".join(batches)+"\n")
    f.write(f"Passes;green,{','.join(passes_list)}\n")
    f.write(f"Fails;red,{','.join(fails_list)}\n")
    f.write(f"Errors;orange,{','.join(errors_list)}\n")
    f.write(f"Timeouts;gray,{','.join(timeout_list)}")
