"""Extract up to N images per class from the HF tanganke/dtd parquet into <dest>/<class>/<k>.jpg."""
import io
import sys
from pathlib import Path

import pyarrow.parquet as pq

parquet, dest, n = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
names = None
pf = pq.ParquetFile(parquet)
import json

meta = json.loads(pf.schema_arrow.metadata[b"huggingface"])["info"]["features"]["label"]["names"]
count = {}
for batch in pf.iter_batches(batch_size=64):
    for img, lab in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist()):
        c = meta[lab]
        k = count.get(c, 0)
        if k >= n:
            continue
        out = dest / c
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{c}_{k:04d}.jpg").write_bytes(img["bytes"])
        count[c] = k + 1
print(f"extracted {sum(count.values())} images, {len(count)} classes -> {dest}")
