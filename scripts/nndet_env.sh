# scripts/nndet_env.sh — nnDetection environment (spec 2026-09-28 brain-nndet §3). Source it with bash; never in the
# same shell as scripts/nnunet_env.sh (nnU-Net v1 inside nnDetection and nnU-Net v2 both read nnUNet_preprocessed).
unset nnUNet_raw nnUNet_preprocessed nnUNet_results nnUNet_raw_data_base RESULTS_FOLDER
export PATH="$HOME/anaconda3/envs/nndet/bin:$PATH"
export PYTHONNOUSERSITE=1
export det_data=/data2/congcong/data/FM_data/derived/nndet
export det_models=/data2/congcong/data/FM_data/derived/nndet_models
export OMP_NUM_THREADS=1
export det_num_threads=8
export det_verbose=1
