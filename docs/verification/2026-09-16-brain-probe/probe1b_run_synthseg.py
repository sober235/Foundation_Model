"""Step 1b: SynthSeg-robust (CPU, nice 19) over each view folder, then labels back to the native grid."""
import os, subprocess, sys, time
from probe_common import *
vols = (OUT / 'volumes.txt').read_text().split()
for view in sys.argv[1:]:
    wd = WORK / view
    (wd / 'seg_1mm').mkdir(exist_ok=True); (wd / 'seg_native').mkdir(exist_ok=True)
    t0 = time.time()
    argv = synthseg_command(in_dir=wd / 'stage', out_dir=wd / 'seg_1mm', resample_dir=None, vol_csv=wd / 'volumes.csv',
                            threads=int(os.environ.get('THREADS', '12')), synthseg_home=Path.home() / 'src/SynthSeg',
                            python=Path.home() / 'anaconda3/envs/synthseg/bin/python', robust=True, cpu=True)
    env = {**os.environ, 'PYTHONNOUSERSITE': '1', 'CUDA_VISIBLE_DEVICES': ''}
    with open(wd / 'synthseg.log', 'a') as log:
        rc = subprocess.run(['nice', '-n', '19'] + argv, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    n = 0
    for stem in vols:
        seg1 = wd / 'seg_1mm' / f'{stem}_synthseg.nii.gz'
        if seg1.exists():
            native_labels(seg1, native_reference(h5_path(stem), wd / 'stage' / f'{stem}.nii.gz'), wd / 'seg_native' / f'{stem}_seg.nii.gz')
            n += 1
    print(f'{view}: rc={rc} native {n}/{len(vols)} in {(time.time() - t0) / 60:.1f} min', flush=True)
