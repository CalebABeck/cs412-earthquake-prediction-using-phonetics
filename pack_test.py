"""Pack 2624 test CSV files into a single .npz for fast loading."""
import os, time
import numpy as np
import pandas as pd

DATA_DIR = "/work/hdd/bfgc/yzhang62/kaggle"
test_dir = os.path.join(DATA_DIR, "test")
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
seg_ids = sub["seg_id"].values

print(f"Packing {len(seg_ids)} test segments ...")
t0 = time.time()
X_test = np.zeros((len(seg_ids), 150_000), dtype=np.int16)

for i, sid in enumerate(seg_ids):
    seg = pd.read_csv(os.path.join(test_dir, f"{sid}.csv"), dtype={"acoustic_data": np.int16})
    X_test[i] = seg["acoustic_data"].values
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(seg_ids)} ... ({time.time()-t0:.0f}s)", flush=True)

out_path = os.path.join(DATA_DIR, "test_data.npz")
np.savez_compressed(out_path, X_test=X_test, seg_ids=seg_ids)
print(f"Done! Saved to {out_path} ({os.path.getsize(out_path)/1e6:.0f} MB) in {time.time()-t0:.0f}s")
