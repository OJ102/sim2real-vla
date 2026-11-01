"""Print/write a JSON GPU snapshot (foreign apps, contended flag). Usage: python scripts/gpu_snapshot.py [out.json]"""
import json
import sys

from vla.utils.gpu import snapshot

s = snapshot()
if len(sys.argv) > 1:
    json.dump(s, open(sys.argv[1], "w"), indent=1)
print(json.dumps(s, indent=1))
