# source scripts/nnunet_env.sh -- paths and limits for every nnU-Net command of the M1 batch
export nnUNet_raw=/data2/congcong/data/FM_data/derived/nnunet/raw
export nnUNet_preprocessed=/data2/congcong/data/FM_data/derived/nnunet/preprocessed
export nnUNet_results=/data2/congcong/data/FM_data/derived/nnunet/results
export nnUNet_n_proc_DA=8
export nnUNet_compile=f
export PYTHONNOUSERSITE=1
export PATH=$HOME/anaconda3/envs/nvgen/bin:$PATH
