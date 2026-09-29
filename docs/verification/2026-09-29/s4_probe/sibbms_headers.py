# Read-only probe of the SibBMS archive: what it holds, and the headers of a few volumes, read from the zip into memory
# (nothing is extracted, nothing is written).
import collections
import gzip
import re
import zipfile

import nibabel as nib
import numpy as np

z = zipfile.ZipFile("/data2/congcong/data/FM_data/SibBMS_ms/sibbms.zip")
names = [n for n in z.namelist() if n.endswith(".nii.gz")]
print("NIfTI files per top-level folder:", dict(collections.Counter(n.split("/")[1] if "/" in n else n for n in names)))
subs = collections.defaultdict(list)
for n in names:
    m = re.search(r"/(MS|Norm)/(sub-\d+)/(ses-\d+)/anat/", "/" + n)
    if m:
        subs[(m.group(1), m.group(2), m.group(3))].append(n)
print("sessions with anat:", len(subs), "| MS", sum(1 for k in subs if k[0] == "MS"), "| Norm", sum(1 for k in subs if k[0] == "Norm"))
print("subjects: MS", len({k[1] for k in subs if k[0] == "MS"}), "| Norm", len({k[1] for k in subs if k[0] == "Norm"}))
kinds = collections.Counter(tuple(sorted(re.sub(r".*_ses-\d+_", "", x).replace(".nii.gz", "") for x in v)) for v in subs.values())
for k, c in kinds.most_common(8):
    print("  sessions", c, "sequences", k)
rng = np.random.default_rng(0)
keys = sorted(subs)
for k in [keys[i] for i in rng.choice(len(keys), 6, replace=False)]:
    for n in sorted(subs[k]):
        img = nib.Nifti1Image.from_bytes(gzip.decompress(z.read(n)))
        a = np.asarray(img.dataobj)
        print(k[0], k[1], re.sub(r".*_ses-\d+_", "", n).replace(".nii.gz", ""), "| shape", img.shape, "| zooms",
              tuple(round(float(x), 3) for x in img.header.get_zooms()[:3]), "| axcodes", "".join(nib.aff2axcodes(img.affine)),
              "| zero fraction %.2f" % float((a == 0).mean()))
