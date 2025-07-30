from pathlib import Path

batches = []
passes = []
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
            if log[-2] == "R":
                errors += 1
            elif log[-2] == "S":
                passes += 1
            elif log[-2] == "L":
                fails += 1
            else:
                # print(f"{iteration.name} did timeout")
                pass

        print(f"{path.name}: {passes} passes, {fails} fails, {errors} errors, {total-passes-errors-fails} "
              f"timeouts/unknown of {total} total")
