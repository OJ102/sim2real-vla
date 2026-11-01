"""Block until the given GPUs are quiet (util<=util-max for hold seconds); on timeout exit 0 with a CONTENDED note.
Usage: python scripts/quiet_wait.py --gpus 0,1,2,4 --util-max 15 --hold 60 --timeout 1200"""
import argparse
import json

from vla.utils.gpu import quiet_wait

ap = argparse.ArgumentParser()
ap.add_argument("--gpus", default="0,1,2,4")
ap.add_argument("--util-max", type=float, default=15)
ap.add_argument("--hold", type=float, default=60)
ap.add_argument("--timeout", type=float, default=1200)
a = ap.parse_args()
print(json.dumps(quiet_wait([int(x) for x in a.gpus.split(",")], a.util_max, a.hold, a.timeout)))
