"""Download the three MaleCNS v1.0 tables the model needs into the current directory.

Anonymous, no token. Writes:
  annotations.parquet  (body-annotations, 13 MB)
  nt.parquet           (body-neurotransmitters, 42 MB)
  weights.feather      (connectome-weights, 1.05 GB, copied as-is; never loaded whole)
Existing files are kept, so a re-run only fetches what is missing.
"""
import os
import pyarrow.feather as pf
import pyarrow.fs as pafs

BASE = "flyem-male-cns/v1.0/connectome-data/flat-connectome/"
gcs = pafs.GcsFileSystem(anonymous=True)

for src, dst in [("body-annotations-male-cns-v1.0-minconf-0.5.feather", "annotations.parquet"),
                 ("body-neurotransmitters-male-cns-v1.0.feather", "nt.parquet")]:
    if not os.path.exists(dst):
        with gcs.open_input_file(BASE + src) as fh:
            pf.read_table(fh).to_pandas().to_parquet(dst)
    print("ok", dst)

if not os.path.exists("weights.feather"):
    pafs.copy_files(BASE + "connectome-weights-male-cns-v1.0-minconf-0.5.feather", "weights.feather",
                    source_filesystem=gcs, destination_filesystem=pafs.LocalFileSystem())
print("ok weights.feather", os.path.getsize("weights.feather"), "bytes")
