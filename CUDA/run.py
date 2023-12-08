import os

for c in ['S', 'W', 'A', 'B', 'C', 'D', 'E']:
    build = f"make clean && make EP CLASS={c}"
    run = f"./bin/ep.{c} > outputs/out_{c}"
    os.system(build)
    os.system(run)

    