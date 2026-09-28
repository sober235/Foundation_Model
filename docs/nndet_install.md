# nnDetection environment install record

Spec: `docs/superpowers/specs/2026-09-28-brain-nndet-design.md`. Env: `~/anaconda3/envs/nndet` (python 3.8, torch
1.11.0+cu113). nnDetection source: `~/src/nnDetection`, detached at `97a58f31`, never modified. The env was installed
by the controller ahead of this branch (done 2026-09-29 02:21); this task did not install, upgrade or remove any
package. Scripts and logs for every run below are in `~/logs/nndet_install/`.

## Install attempts (Task 1 Step 5)

Seven runs were needed to reach a working build.

### 01_env.sh — the README route (abandoned)

```bash
# Plan 2026-09-28-brain-nndet, Task 1 Step 5 (run by the controller ahead of the branch; commands verbatim).
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897
~/anaconda3/bin/conda create -y -n nndet python=3.8
source ~/anaconda3/bin/activate nndet
conda install -y gxx_linux-64==9.3.0
conda install -y cuda -c nvidia/label/cuda-11.3.1
conda install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
cd ~/src/nnDetection
git checkout --detach 97a58f31
export PYTHONNOUSERSITE=1
pip install -r requirements.txt
pip install hydra-core --upgrade --pre
pip install git+https://github.com/mibaumgartner/pytorch_model_summary.git
pip install pytest
CUDA_HOME=$CONDA_PREFIX FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.0 pip install -v -e .
git status --short --untracked-files=no
echo "ENV_SETUP_DONE $(date '+%F %T')"
```

Last 15 lines of `01_env.log`:

```
++ '[' -n '' ']'
++ hash -r
+ conda install -y gxx_linux-64==9.3.0
+ local cmd=install
+ case "$cmd" in
+ __conda_exe install -y gxx_linux-64==9.3.0
+ /home/congcongliu/anaconda3/bin/conda install -y gxx_linux-64==9.3.0
Collecting package metadata (current_repodata.json): ...working... done
Solving environment: ...working... failed with initial frozen solve. Retrying with flexible solve.
Collecting package metadata (repodata.json): ...working... WARNING conda.models.version:get_matcher(546): Using .* with relational operator is superfluous and deprecated and will be removed in a future version of conda. Your spec was 1.9.0.*, but conda is ignoring the .* and treating it as 1.9.0
WARNING conda.models.version:get_matcher(546): Using .* with relational operator is superfluous and deprecated and will be removed in a future version of conda. Your spec was 1.8.0.*, but conda is ignoring the .* and treating it as 1.8.0
WARNING conda.models.version:get_matcher(546): Using .* with relational operator is superfluous and deprecated and will be removed in a future version of conda. Your spec was 1.7.1.*, but conda is ignoring the .* and treating it as 1.7.1
WARNING conda.models.version:get_matcher(546): Using .* with relational operator is superfluous and deprecated and will be removed in a future version of conda. Your spec was 1.6.0.*, but conda is ignoring the .* and treating it as 1.6.0
Terminated
+ return
```

Abandoned: the classic conda solver ran 28 minutes on `gxx_linux-64==9.3.0` against the full Tsinghua conda-forge
repodata (`~/.condarc`: Tsinghua mirror, strict channel priority) and was stopped.

### 02_env.sh — override-channels compiler route (abandoned)

```bash
# Plan 2026-09-28-brain-nndet, Task 1 Step 5, second route (2026-09-29 00:10): the classic solver hung for 28 min on
# gxx_linux-64 against the full Tsinghua conda-forge repodata, so each conda install is restricted to the one channel
# that holds the package (--override-channels), and torch comes from the official cu113 pip wheel. Versions unchanged:
# gxx 9.3.0, CUDA 11.3.1 (nvcc), torch 1.11.0 + cu113, torchvision 0.12.0, torchaudio 0.11.0. The env already exists
# (01_env.sh created it: python 3.8.20).
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897
source ~/anaconda3/bin/activate nndet
conda install -y --override-channels -c defaults gxx_linux-64=9.3.0
conda install -y --override-channels -c nvidia/label/cuda-11.3.1 cuda
export PYTHONNOUSERSITE=1
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
cd ~/src/nnDetection
git checkout --detach 97a58f31
pip install -r requirements.txt
pip install hydra-core --upgrade --pre
pip install git+https://github.com/mibaumgartner/pytorch_model_summary.git
pip install pytest
python -c "import torch; print('torch', torch.__version__, torch.version.cuda)"
which nvcc; nvcc -V | tail -2; x86_64-conda-linux-gnu-g++ --version | head -1
CUDA_HOME=$CONDA_PREFIX FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.0 pip install -v -e .
git status --short --untracked-files=no
echo "ENV_SETUP_DONE $(date '+%F %T')"
```

Last 15 lines of `02_env.log`:

```
  - wheel -> python[version='>=3.14,<3.15.0a0'] -> __glibc[version='>=2.17,<3.0.a0|>=2.28,<3.0.a0']
  - xz -> __glibc[version='>=2.17,<3.0.a0|>=2.28,<3.0.a0']
  - xz -> libgcc-ng[version='>=11.2.0'] -> __glibc[version='>=2.17']
  - xz-gpl-tools -> __glibc[version='>=2.17,<3.0.a0']
  - xz-gpl-tools -> libgcc[version='>=14'] -> __glibc[version='>=2.28,<3.0.a0']
  - xz-tools -> __glibc[version='>=2.17,<3.0.a0']
  - xz-tools -> libgcc[version='>=14'] -> __glibc[version='>=2.28,<3.0.a0']
  - zstd -> __glibc[version='>=2.17,<3.0.a0|>=2.28,<3.0.a0']
  - zstd -> libgcc-ng[version='>=11.2.0'] -> __glibc[version='>=2.17']

Your installed version is: 2.35

Note that strict channel priority may have removed packages required for satisfiability.

+ return
```

Abandoned: `defaults`' `gxx_linux-64=9.3.0` is unsatisfiable against the env's conda-forge python/libgcc under strict
channel priority (glibc version conflict).

### 03_env.sh — no conda compiler, system gcc-10 (abandoned)

```bash
# Plan 2026-09-28-brain-nndet, Task 1 Step 5, third route (2026-09-29): no conda compiler. The env's python/libgcc
# come from conda-forge (strict channel priority), so defaults' gxx_linux-64 9.3.0 is unsatisfiable (02_env.log) and
# the mixed-channel solve hung (01_env.log). The system has gcc-10/g++-10, which nvcc 11.3 supports (host compiler
# <= 10), so the extension is built with CC=gcc-10 CXX=g++-10. nvcc 11.3.1 comes from the nvidia label channel alone;
# torch 1.11.0+cu113 from the official pip wheel. Env created by 01_env.sh (python 3.8.20).
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897
source ~/anaconda3/bin/activate nndet
conda install -y --override-channels -c nvidia/label/cuda-11.3.1 cuda
export PYTHONNOUSERSITE=1
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
cd ~/src/nnDetection
git checkout --detach 97a58f31
pip install -r requirements.txt
pip install hydra-core --upgrade --pre
pip install git+https://github.com/mibaumgartner/pytorch_model_summary.git
pip install pytest
python -c "import torch; print('torch', torch.__version__, torch.version.cuda)"
which nvcc; nvcc -V | tail -2; gcc-10 --version | head -1; g++-10 --version | head -1
CC=gcc-10 CXX=g++-10 CUDA_HOME=$CONDA_PREFIX FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.0 pip install -v -e .
git status --short --untracked-files=no
echo "ENV_SETUP_DONE $(date '+%F %T')"
```

Last 15 lines of `03_env.log`:

```
++ export CONDA_SHLVL=2
++ CONDA_SHLVL=2
++ export 'CONDA_PROMPT_MODIFIER=(nndet) '
++ CONDA_PROMPT_MODIFIER='(nndet) '
+ __conda_hashr
+ '[' -n '' ']'
+ '[' -n '' ']'
+ hash -r
+ export PYTHONNOUSERSITE=1
+ PYTHONNOUSERSITE=1
+ pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple, https://download.pytorch.org/whl/cu113
Collecting torch==1.11.0+cu113
  Downloading https://download-r2.pytorch.org/whl/cu113/torch-1.11.0%2Bcu113-cp310-cp310-linux_x86_64.whl (1637.0 MB)
Terminated
```

Abandoned: the conda step succeeded (nvcc 11.3.122 installed in the env), but the bare `pip` resolved to
`/usr/bin/pip` (system python 3.10, wrong wheel tag `cp310`) because `conda activate` puts the env's `bin` after
`~/.local/bin` and `/usr/bin` in this user's `PATH`. Stopped while downloading; nothing was installed.

### 04_env.sh — absolute env python, still proxy download (abandoned)

```bash
# Plan 2026-09-28-brain-nndet, Task 1 Step 5, fourth run (2026-09-29): continues 03_env.sh after its conda step
# (nvcc 11.3.122 is in the env). In 03 the bare `pip` resolved to /usr/bin/pip (system python 3.10): conda activate
# put the env's bin after ~/.local/bin and /usr/bin in this user's PATH. It was stopped while downloading; nothing was
# installed (no "Installing collected packages"; nothing in ~/.local newer than 03_env.sh). Every command here names
# the env's python by absolute path and PATH is prepended explicitly.
E=$HOME/anaconda3/envs/nndet
export PATH="$E/bin:$PATH"
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897
export PYTHONNOUSERSITE=1
"$E/bin/python" -c "import sys; assert sys.version_info[:2] == (3, 8), sys.version; print(sys.executable)"
"$E/bin/python" -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
cd ~/src/nnDetection
git checkout --detach 97a58f31
"$E/bin/python" -m pip install -r requirements.txt
"$E/bin/python" -m pip install hydra-core --upgrade --pre
"$E/bin/python" -m pip install git+https://github.com/mibaumgartner/pytorch_model_summary.git
"$E/bin/python" -m pip install pytest
"$E/bin/python" -c "import sys, torch; print(sys.executable, 'torch', torch.__version__, torch.version.cuda)"
"$E/bin/nvcc" -V | tail -2; gcc-10 --version | head -1; g++-10 --version | head -1
CC=gcc-10 CXX=g++-10 CUDA_HOME="$E" FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.0 "$E/bin/python" -m pip install -v -e .
git status --short --untracked-files=no
echo "ENV_SETUP_DONE $(date '+%F %T')"
```

Last 15 lines of `04_env.log`:

```
+ http_proxy=http://127.0.0.1:7897
+ export PYTHONNOUSERSITE=1
+ PYTHONNOUSERSITE=1
+ /home/congcongliu/anaconda3/envs/nndet/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 8), sys.version; print(sys.executable)'
/home/congcongliu/anaconda3/envs/nndet/bin/python
+ /home/congcongliu/anaconda3/envs/nndet/bin/python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple, https://download.pytorch.org/whl/cu113
Collecting torch==1.11.0+cu113
  Downloading https://download-r2.pytorch.org/whl/cu113/torch-1.11.0%2Bcu113-cp38-cp38-linux_x86_64.whl (1637.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━                  0.9/1.6 GB 586.2 kB/s eta 0:19:46
ERROR: THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE. If you have updated the package versions, please update the hashes. Otherwise, examine the package contents carefully; someone may have tampered with them.
    torch==1.11.0+cu113 from https://download-r2.pytorch.org/whl/cu113/torch-1.11.0%2Bcu113-cp38-cp38-linux_x86_64.whl#sha256=b6a799bdb6ee3d914e5e62bddb4276d4a10248c1af4f2d217738e5f9ee27485b:
        Expected sha256 b6a799bdb6ee3d914e5e62bddb4276d4a10248c1af4f2d217738e5f9ee27485b
             Got        94342a15a1b67e7e8e3e305748f8505e473f2e0fb7eebc4631e25cc405c2360c

```

Abandoned: the 1.6 GB torch wheel download through the proxy (~0.55 MB/s single connection) was cut at 0.9 GB and
failed its sha256 check.

### 05_env.sh — aria2c fetch, blocked by mirror (abandoned)

```bash
# Plan 2026-09-28-brain-nndet, Task 1 Step 5, fifth run (2026-09-29): 04_env.sh's pip download of the 1.6 GB torch
# wheel was cut at 0.9 GB (~0.55 MB/s single connection) and failed its sha256 check. The wheel is now fetched with
# aria2c (8 connections, resumable) from the Aliyun PyTorch mirror and checked against the official sha256 from
# download.pytorch.org (b6a799bd...); pip then installs the verified local file. Everything else as 04_env.sh.
E=$HOME/anaconda3/envs/nndet
W=$HOME/logs/nndet_install/wheels
WHL=torch-1.11.0+cu113-cp38-cp38-linux_x86_64.whl
SHA=b6a799bdb6ee3d914e5e62bddb4276d4a10248c1af4f2d217738e5f9ee27485b
export PATH="$E/bin:$PATH"
export PYTHONNOUSERSITE=1
"$E/bin/python" -c "import sys; assert sys.version_info[:2] == (3, 8), sys.version; print(sys.executable)"
mkdir -p "$W"
aria2c -x 8 -s 8 -c --checksum=sha-256=$SHA --summary-interval=60 -d "$W" -o "$WHL" "https://mirrors.aliyun.com/pytorch-wheels/cu113/$WHL"
echo "$SHA  $W/$WHL" | sha256sum -c -
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897
"$E/bin/python" -m pip install "$W/$WHL" torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
cd ~/src/nnDetection
git checkout --detach 97a58f31
"$E/bin/python" -m pip install -r requirements.txt
"$E/bin/python" -m pip install hydra-core --upgrade --pre
"$E/bin/python" -m pip install git+https://github.com/mibaumgartner/pytorch_model_summary.git
"$E/bin/python" -m pip install pytest
"$E/bin/python" -c "import sys, torch; print(sys.executable, 'torch', torch.__version__, torch.version.cuda)"
"$E/bin/nvcc" -V | tail -2; gcc-10 --version | head -1; g++-10 --version | head -1
CC=gcc-10 CXX=g++-10 CUDA_HOME="$E" FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.0 "$E/bin/python" -m pip install -v -e .
git status --short --untracked-files=no
echo "ENV_SETUP_DONE $(date '+%F %T')"
```

Last 15 lines of `05_env.log`:

```
Exception: [AbstractCommand.cc:351] errorCode=22 URI=https://mirrors.aliyun.com/pytorch-wheels/cu113/torch-1.11.0+cu113-cp38-cp38-linux_x86_64.whl
  -> [HttpSkipResponseCommand.cc:239] errorCode=22 The response status is not successful. status=403

09/29 01:52:05 [1;32mNOTICE[0m] Download GID#6c44234a17dfba32 not complete: /home/congcongliu/logs/nndet_install/wheels/torch-1.11.0+cu113-cp38-cp38-linux_x86_64.whl

Download Results:
gid   |stat|avg speed  |path/URI
======+====+===========+=======================================================
6c4423|ERR |       0B/s|/home/congcongliu/logs/nndet_install/wheels/torch-1.11.0+cu113-cp38-cp38-linux_x86_64.whl

Status Legend:
(ERR):error occurred.

aria2 will resume download if the transfer is restarted.
If there are any errors, then see the log file. See '-l' option in help/man page for details.
```

Abandoned: aria2c got HTTP 403 from the Aliyun mirror using its default user agent.

### 06_env.sh — aria2c with curl user agent, build fails on setuptools (abandoned)

```bash
# Plan 2026-09-28-brain-nndet, Task 1 Step 5, sixth run (the fifth got HTTP 403 from the mirror for aria2c's default user agent; -U curl/7.81.0 passes) (2026-09-29): 04_env.sh's pip download of the 1.6 GB torch
# wheel was cut at 0.9 GB (~0.55 MB/s single connection) and failed its sha256 check. The wheel is now fetched with
# aria2c (8 connections, resumable) from the Aliyun PyTorch mirror and checked against the official sha256 from
# download.pytorch.org (b6a799bd...); pip then installs the verified local file. Everything else as 04_env.sh.
E=$HOME/anaconda3/envs/nndet
W=$HOME/logs/nndet_install/wheels
WHL=torch-1.11.0+cu113-cp38-cp38-linux_x86_64.whl
SHA=b6a799bdb6ee3d914e5e62bddb4276d4a10248c1af4f2d217738e5f9ee27485b
export PATH="$E/bin:$PATH"
export PYTHONNOUSERSITE=1
"$E/bin/python" -c "import sys; assert sys.version_info[:2] == (3, 8), sys.version; print(sys.executable)"
mkdir -p "$W"
aria2c -U curl/7.81.0 -x 8 -s 8 -c --checksum=sha-256=$SHA --summary-interval=60 -d "$W" -o "$WHL" "https://mirrors.aliyun.com/pytorch-wheels/cu113/$WHL"
echo "$SHA  $W/$WHL" | sha256sum -c -
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897
"$E/bin/python" -m pip install "$W/$WHL" torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
cd ~/src/nnDetection
git checkout --detach 97a58f31
"$E/bin/python" -m pip install -r requirements.txt
"$E/bin/python" -m pip install hydra-core --upgrade --pre
"$E/bin/python" -m pip install git+https://github.com/mibaumgartner/pytorch_model_summary.git
"$E/bin/python" -m pip install pytest
"$E/bin/python" -c "import sys, torch; print(sys.executable, 'torch', torch.__version__, torch.version.cuda)"
"$E/bin/nvcc" -V | tail -2; gcc-10 --version | head -1; g++-10 --version | head -1
CC=gcc-10 CXX=g++-10 CUDA_HOME="$E" FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.0 "$E/bin/python" -m pip install -v -e .
git status --short --untracked-files=no
echo "ENV_SETUP_DONE $(date '+%F %T')"
```

The wheel downloaded and checked out this time (`sha256sum -c -` OK) and `import torch` printed
`/home/congcongliu/anaconda3/envs/nndet/bin/python torch 1.11.0+cu113 11.3`. Last 15 lines of `06_env.log`:

```
  else:
      filename = "<auto-generated setuptools caller>"
      setup_py_code = "from setuptools import setup; setup()"
  
  exec(compile(setup_py_code, filename, "exec"))
  '"'"''"'"''"'"' % ('"'"'/home/congcongliu/src/nnDetection/setup.py'"'"',), "<pip-setuptools-caller>", "exec"))' egg_info --egg-base /tmp/pip-pip-egg-info-ps0rtrc6
  cwd: /home/congcongliu/src/nnDetection/
  Preparing metadata (setup.py): finished with status 'error'
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> See above for output.

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.
```

The underlying error, a few hundred lines earlier in the same log:

```
      from pkg_resources import packaging  # type: ignore[attr-defined]
  ImportError: cannot import name 'packaging' from 'pkg_resources' (/home/congcongliu/anaconda3/envs/nndet/lib/python3.8/site-packages/pkg_resources/__init__.py)
```

Abandoned (only the final build step): torch 1.11's `cpp_extension` needs an older setuptools than the one pip
resolved (setuptools 70.3.0 is too new — `pkg_resources.packaging` was removed).

### 07_build.sh — working finish (succeeded)

```bash
# Plan 2026-09-28-brain-nndet, Task 1 Step 5, seventh run (2026-09-29): 06_env.sh installed torch 1.11.0+cu113
# (sha256 verified), requirements, hydra, pytorch_model_summary and pytest; its last step (building nnDetection's
# extension) failed with "cannot import name 'packaging' from 'pkg_resources'": torch 1.11's cpp_extension needs an old
# setuptools. setuptools is pinned to 59.5.0 (the version the plan's Task 1 Step 6 names for the related
# distutils.version error) and only the build step is repeated.
E=$HOME/anaconda3/envs/nndet
export PATH="$E/bin:$PATH"
export PYTHONNOUSERSITE=1
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897
"$E/bin/python" -c "import sys; assert sys.version_info[:2] == (3, 8), sys.version; print(sys.executable)"
"$E/bin/python" -m pip install setuptools==59.5.0
"$E/bin/python" -c "import setuptools, torch; print('setuptools', setuptools.__version__, 'torch', torch.__version__)"
cd ~/src/nnDetection
git rev-parse --short HEAD
CC=gcc-10 CXX=g++-10 CUDA_HOME="$E" FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.0 "$E/bin/python" -m pip install -v -e .
git status --short --untracked-files=no
echo "ENV_SETUP_DONE $(date '+%F %T')"
```

Last 15 lines of `07_build.log`:

```
    Installing nndet_predict script to /home/congcongliu/anaconda3/envs/nndet/bin
    Installing nndet_prep script to /home/congcongliu/anaconda3/envs/nndet/bin
    Installing nndet_searchpath script to /home/congcongliu/anaconda3/envs/nndet/bin
    Installing nndet_seg2det script to /home/congcongliu/anaconda3/envs/nndet/bin
    Installing nndet_seg2nii script to /home/congcongliu/anaconda3/envs/nndet/bin
    Installing nndet_sweep script to /home/congcongliu/anaconda3/envs/nndet/bin
    Installing nndet_train script to /home/congcongliu/anaconda3/envs/nndet/bin
    Installing nndet_unpack script to /home/congcongliu/anaconda3/envs/nndet/bin

    Installed /home/congcongliu/src/nnDetection
Successfully installed argparse-1.4.0 nndet-0.1
+ git status --short --untracked-files=no
++ date '+%F %T'
+ echo 'ENV_SETUP_DONE 2026-09-29 02:21:04'
ENV_SETUP_DONE 2026-09-29 02:21:04
```

Succeeded: `git status --short --untracked-files=no` printed nothing (the nnDetection worktree stayed clean at
`97a58f31`) and the sentinel line above closed the run.

### Precondition check (this task, before doing any work)

```
$ grep ENV_SETUP_DONE ~/logs/nndet_install/07_build.log
+ echo 'ENV_SETUP_DONE 2026-09-29 02:21:04'
ENV_SETUP_DONE 2026-09-29 02:21:04
$ bash -c 'cd ~/src/nnDetection && git rev-parse --short HEAD && git status --short --untracked-files=no'
97a58f31
```

Both matched the brief's expectation, so the task proceeded.

## Step 6: import check

Idle GPUs were selected with:

```
$ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
0, 14
1, 28159
2, 14
3, 14
4, 14
5, 26351
6, 25869
7, 25755
$ nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader
GPU-05cb1334-b9c8-1afc-ec78-b0eef86e193d, 1903554
GPU-18b3d14a-7ef2-870d-d438-1d6d09755a2d, 2586011
GPU-61e060ef-8507-bae3-7db9-2a4fef8d408a, 2586013
GPU-20a60eab-8897-efa2-5c04-ce8d2e284218, 2586015
```

GPUs 0, 2, 3 and 4 had < 1000 MiB used and no compute app (the busy GPUs 1, 5, 6, 7 map to the four listed compute-app
UUIDs). GPU 0 was used.

Command:

```
bash -c 'source scripts/nndet_env.sh && cd /tmp && CUDA_VISIBLE_DEVICES=0 python -c "import torch, nndet._C, nndet; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"'
```

Output:

```
1.11.0+cu113 11.3 True
```

No `distutils.version` error occurred, so the setuptools fallback in the brief was not needed.

## Step 7: toy smoke run

Smoke root did not exist beforehand (`test ! -e /data2/congcong/data/FM_data/derived/nndet_smoke` passed). Re-checked
idle GPUs immediately before launch — same result as Step 6 (0, 2, 3, 4 idle) — and used GPU 0 again.

Command:

```bash
bash -ex <<'EOF' > logs/nndet_install/02_smoke.log 2>&1
source scripts/nndet_env.sh
export det_data=/data2/congcong/data/FM_data/derived/nndet_smoke/data
export det_models=/data2/congcong/data/FM_data/derived/nndet_smoke/models
mkdir -p "$det_data" "$det_models"
cd "$det_data"
nndet_example
nice -n 19 nndet_prep 000 -np 4 -npp 4
nice -n 19 nndet_unpack "$det_data/Task000D3_Example/preprocessed/D3V001_3d/imagesTr" 4
CUDA_VISIBLE_DEVICES=0 nice -n 19 nndet_train 000 -o train=smoke --sweep
EOF
```

Run in the background (subject to a 90-minute cap per the task brief). `train.log` shows all real work finished at
**02:41:37** (training, prediction, sweep and both analysis suites all completed; `model_last.ckpt`'s mtime is
02:40:36, before the sweep). After that, the process did not exit: 6 threads sat in `futex`/`poll` wait holding
~2.7 GB on GPU 0, and stderr carried a `batchgenerators` teardown exception —

```
Exception in thread Thread-1:
Traceback (most recent call last):
  File "/home/congcongliu/anaconda3/envs/nndet/lib/python3.8/threading.py", line 932, in _bootstrap_inner
    self.run()
  File "/home/congcongliu/anaconda3/envs/nndet/lib/python3.8/threading.py", line 870, in run
    self._target(*self._args, **self._kwargs)
  File "/home/congcongliu/anaconda3/envs/nndet/lib/python3.8/site-packages/batchgenerators/dataloading/multi_threaded_augmenter.py", line 98, in results_loop
    raise RuntimeError("One or more background workers are no longer alive. Exiting. Please check the print"
RuntimeError: One or more background workers are no longer alive. Exiting. Please check the print statements above for the actual error message
```

— raised in a side thread (`MultiThreadedAugmenter`'s watcher) at the shutdown of the last analysis step, after its
progress bar had already reached 100%. It did not propagate to the main process, which stayed alive but never
returned. All expected output files were already complete on disk at that point (confirmed below). The controller
terminated the hung process with SIGTERM at about 03:02, which is why the smoke script's own exit code is 143
(`echo "EXIT_CODE=$?" >> logs/nndet_install/02_smoke.log` recorded `EXIT_CODE=143`). This is a known teardown quirk,
not a training or sweep failure, and nothing was changed to work around it.

### Confirmations

```
$ T=/data2/congcong/data/FM_data/derived/nndet_smoke/models/Task000D3_Example/RetinaUNetV001_D3V001_3d/fold0
$ ls $T
config_resolved.yaml
config.yaml
meta.json
mlruns
model_best.ckpt
model_last.ckpt
network.txt
plan_debug.json
plan_inference.pkl
plan.pkl
requirements.txt
splits.pkl
sweep
sweep_predictions
train.log
val_analysis
val_analysis_preprocessed
val_predictions
val_predictions_preprocessed
val_results
val_results_preprocessed
$ ls $T/sweep_predictions | head
case_0_boxes.pt
case_0_properties.pkl
case_7_boxes.pt
case_7_properties.pkl
$ ls $T/sweep_predictions/*_boxes.pt | wc -l
2
```

`plan.pkl` / `plan_inference.pkl`:

```
$ bash -c "source scripts/nndet_env.sh && python -c \"import pickle; p = pickle.load(open('$T/plan.pkl', 'rb')); print('inference_plan' in p, p['data_identifier'], p['planner_id'], p['transpose_backward']); q = pickle.load(open('$T/plan_inference.pkl', 'rb')); print(sorted(q['inference_plan']))\""
False D3V001_3d D3V001 [0, 1, 2]
['ensemble_iou', 'ensemble_nms_fn', 'ensemble_score_thresh', 'ensemble_topk', 'model_detections_per_image', 'model_iou', 'model_nms_fn', 'model_score_thresh', 'model_topk', 'remove_small_boxes']
```

So `plan.pkl` holds no `inference_plan` key (that lives in `plan_inference.pkl`), the data identifier is `D3V001_3d`,
planner id `D3V001`, transpose-backward `[0, 1, 2]`; the 10 swept parameters are the ones listed above (from
`BoxEnsemblerSelective.get_default_parameters()`).

`train.log` markers and timestamps:

```
$ grep -n -E 'Predict cases with default settings|Start parameter sweep|Found inference plan' $T/train.log
62:2026-09-29 02:40:37.001 | INFO ... sweep:788 - Predict cases with default settings...
82:2026-09-29 02:40:37.368 | INFO ... get_predictor:710 - Found inference plan: {} for prediction
93:2026-09-29 02:41:15.619 | INFO ... sweep:803 - Start parameter sweep...
```

`train.log`'s last line is timestamped 2026-09-29 02:41:37.558 (matches the file's mtime,
`2026-09-29 02:41:37.558188190 +0800`), i.e. the run's real work (predict + sweep + both analysis suites) took
**60.557 s** wall time from "Predict cases with default settings..." (02:40:37.001) to the end of the log
(02:41:37.558). It predicted **2** validation cases (`case_0`, `case_7` — same two cases in `sweep_predictions`,
`val_predictions` and `val_predictions_preprocessed`).

## Environment freeze

```
$ ~/anaconda3/envs/nndet/bin/nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Wed_Jun__9_20:32:38_PDT_2021
Cuda compilation tools, release 11.3, V11.3.122
Build cuda_11.3.r11.3/compiler.30059648_0
```

```
$ g++-10 --version | head -1
g++-10 (Ubuntu 10.5.0-1ubuntu1~22.04) 10.5.0
```

```
$ bash -c 'source scripts/nndet_env.sh && pip freeze'
absl-py==2.3.1
aiohappyeyeballs==2.4.4
aiohttp==3.10.11
aiosignal==1.3.1
alembic==1.14.1
antlr4-python3-runtime==4.9.3
async-timeout==5.0.1
attrs==25.3.0
batchgenerators==0.25.3
bayesian-optimization==1.4.0
blinker==1.8.2
cachetools==5.5.2
certifi==2026.7.22
cffi==1.17.1
charset-normalizer==3.5.1
click==8.1.8
cloudpickle==3.1.2
cma==4.5.0
contourpy==1.1.1
cryptography==47.0.0
cycler==0.12.1
databricks-sdk==0.102.0
Deprecated==1.3.1
dicom2nifti==2.6.0
directsearch==1.0
docker==7.2.0
exceptiongroup==1.3.1
Flask==3.0.3
fonttools==4.57.0
frozenlist==1.5.0
fsspec==2025.3.0
future==1.0.0
gitdb==4.0.12
GitPython==3.1.62
google-auth==2.50.0
google-auth-oauthlib==1.0.0
graphene==3.4.3
graphql-core==3.2.13
graphql-relay==3.2.0
greenlet==3.1.1
grpcio==1.70.0
gunicorn==23.0.0
hydra-core==1.4.0.dev1
idna==3.15
imageio==2.35.1
importlib_metadata==8.5.0
importlib_resources==6.4.5
iniconfig==2.1.0
itsdangerous==2.2.0
Jinja2==3.1.6
joblib==1.4.2
kiwisolver==1.4.7
lazy_loader==0.4
linecache2==1.0.0
loguru==0.7.3
Mako==1.3.12
Markdown==3.7
MarkupSafe==2.1.5
matplotlib==3.7.5
MedPy==0.4.0
mlflow==2.17.2
mlflow-skinny==2.17.2
multidict==6.1.0
networkx==3.1
nevergrad==1.0.12
nibabel==5.2.1
-e git+https://github.com/MIC-DKFZ/nnDetection.git@97a58f3110b71caf1b4bcc1851e67cf11e987fc5#egg=nndet
nnunet==1.7.1
numpy==1.24.4
oauthlib==3.3.1
omegaconf==2.4.0.dev3
opentelemetry-api==1.33.1
opentelemetry-sdk==1.33.1
opentelemetry-semantic-conventions==0.54b1
packaging==24.2
pandas==2.0.3
pillow==10.4.0
pluggy==1.5.0
propcache==0.2.0
protobuf==5.29.6
pyarrow==17.0.0
pyasn1==0.6.4
pyasn1_modules==0.4.2
pycparser==2.23
pyDeprecate==0.3.1
pydicom==2.4.4
pyparsing==3.1.4
pytest==8.3.5
python-dateutil==2.9.0.post0
python-gdcm==3.0.25
pytorch-lightning==1.4.2
pytorch_model_summary @ git+https://github.com/mibaumgartner/pytorch_model_summary.git@4045942cd3e473dd7cdeda6431848967d851238f
pytz==2026.4
PyWavelets==1.4.1
PyYAML==6.0.3
requests==2.32.4
requests-oauthlib==2.0.0
scikit-image==0.21.0
scikit-learn==1.3.2
scipy==1.10.1
seaborn==0.13.2
SimpleITK==2.0.2
six==1.17.0
smmap==5.0.3
SQLAlchemy==2.0.54
sqlparse==0.5.5
tensorboard==2.14.0
tensorboard-data-server==0.7.2
threadpoolctl==3.5.0
tifffile==2023.7.10
tomli==2.4.1
torch @ file:///home/congcongliu/logs/nndet_install/wheels/torch-1.11.0%2Bcu113-cp38-cp38-linux_x86_64.whl
torchaudio==0.11.0+cu113
torchmetrics==0.7.3
torchvision==0.12.0+cu113
tqdm==4.70.1
traceback2==1.4.0
typing_extensions==4.13.2
tzdata==2026.4
unittest2==1.1.0
urllib3==2.2.3
Werkzeug==3.0.6
wrapt==2.0.1
yarl==1.15.2
zipp==3.20.2
```

## Runner on the toy task

Task 3's `scripts/nndet_runner.py` was checked against Task 1's toy smoke outputs (`Task000D3_Example`, fold 0).
The runner output directory did not exist beforehand
(`test ! -e /data2/congcong/data/FM_data/derived/nndet_smoke/runner` passed). Idle GPUs were re-checked immediately
before the `predict` call:

```
$ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
0, 14
1, 8453
2, 14
3, 14
4, 14
5, 26351
6, 25869
7, 25755
$ nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader
GPU-05cb1334-b9c8-1afc-ec78-b0eef86e193d, 3054677
GPU-18b3d14a-7ef2-870d-d438-1d6d09755a2d, 2586011
GPU-61e060ef-8507-bae3-7db9-2a4fef8d408a, 2586013
GPU-20a60eab-8897-efa2-5c04-ce8d2e284218, 2586015
```

GPUs 0, 2, 3 and 4 had < 1000 MiB used and no compute app; GPU 2 was used.

Command:

```bash
bash -ex <<'EOF' > logs/nndet_install/03_runner_toy.log 2>&1
source scripts/nndet_env.sh
export det_data=/data2/congcong/data/FM_data/derived/nndet_smoke/data
export det_models=/data2/congcong/data/FM_data/derived/nndet_smoke/models
T=$det_models/Task000D3_Example/RetinaUNetV001_D3V001_3d/fold0
W=/data2/congcong/data/FM_data/derived/nndet_smoke/runner
python scripts/nndet_runner.py extract --state $T/sweep_predictions --params default --out $W/extract_default.json
python scripts/nndet_runner.py extract --state $T/sweep_predictions --params swept --train-dir $T --out $W/extract_swept.json
python scripts/nndet_runner.py gt --prep $det_data/Task000D3_Example/preprocessed --out $W/gt.json
IMG=$(ls $det_data/Task000D3_Example/raw_splitted/imagesTs/*_0000.nii.gz | head -1)
CUDA_VISIBLE_DEVICES=2 python scripts/nndet_runner.py predict --image $IMG --train-dir $T --work $W/predict_work --out $W/predict.json
EOF
```

The run finished on its own (exit code 0, no hung process this time). Its four `wrote ...` lines:

```
wrote /data2/congcong/data/FM_data/derived/nndet_smoke/runner/extract_default.json: 2 cases, 1575 boxes
wrote /data2/congcong/data/FM_data/derived/nndet_smoke/runner/extract_swept.json: 2 cases, 396 boxes
wrote /data2/congcong/data/FM_data/derived/nndet_smoke/runner/gt.json: 10 cases, 10 boxes
wrote /data2/congcong/data/FM_data/derived/nndet_smoke/runner/predict.json: 1 cases, 789 boxes
```

`grep -n "Found inference plan" logs/nndet_install/03_runner_toy.log`:

```
55:2026-09-29 03:15:19.111 | INFO     | nndet.ptmodule.retinaunet.base:get_predictor:710 - Found inference plan: {} for prediction
```

Check against nnDetection's own outputs (nndet env):

```bash
bash -c 'source scripts/nndet_env.sh && python - <<"PY"
import json, pickle
from pathlib import Path
import numpy as np
import SimpleITK as sitk
D = Path("/data2/congcong/data/FM_data/derived/nndet_smoke/data/Task000D3_Example")
T = Path("/data2/congcong/data/FM_data/derived/nndet_smoke/models/Task000D3_Example/RetinaUNetV001_D3V001_3d/fold0")
W = Path("/data2/congcong/data/FM_data/derived/nndet_smoke/runner")
plan = pickle.load(open(D / "preprocessed/D3V001_3d.pkl", "rb"))
stage = D / "preprocessed" / plan["data_identifier"] / "imagesTr"
tb = list(plan["transpose_backward"])
# 1) swept extraction == nnDetection val_predictions (restored), high ends equal, low ends shifted by the spacing ratio
sw = json.load(open(W / "extract_swept.json"))["cases"]
worst = 0.0
for case, rec in sw.items():
    theirs = pickle.load(open(T / "val_predictions" / f"{case}_boxes.pkl", "rb"))
    tbx = np.asarray(theirs["pred_boxes"], float).reshape(-1, 6)
    ours = np.asarray(rec["boxes"], float).reshape(-1, 6)
    props = pickle.load(open(stage / f"{case}.pkl", "rb"))
    scale = np.asarray(props["spacing_after_resampling"])[tb] / np.asarray(props["original_spacing"])
    assert ours.shape == tbx.shape and np.allclose(rec["scores"], np.asarray(theirs["pred_scores"]).reshape(-1))
    assert np.allclose(ours[:, [2, 3, 5]], tbx[:, [2, 3, 5]])
    assert np.allclose(ours[:, [0, 1, 4]] - tbx[:, [0, 1, 4]], scale[[0, 1, 2]])
    worst = max(worst, float(np.abs(ours - tbx).max()) if len(ours) else 0.0)
print("swept vs val_predictions: cases", len(sw), "all consistent; max abs diff", worst)
# 2) gt conversion == tight boxes of the raw instance labels (exact when the case was not resampled)
gt = json.load(open(W / "gt.json"))["cases"]
n, errs = 0, []
for case, rec in gt.items():
    lab = sitk.GetArrayFromImage(sitk.ReadImage(str(D / "raw_splitted/labelsTr" / f"{case}.nii.gz")))
    for k, b in zip(rec["instances"], rec["boxes"]):
        s, r, c = np.nonzero(lab == k)
        tight = np.array([s.min(), r.min(), s.max() + 1, r.max() + 1, c.min(), c.max() + 1], float)
        errs.append(float(np.abs(np.asarray(b) - tight).max())); n += 1
print("gt instances", n, "max coordinate error", max(errs), "exact", sum(e < 1e-6 for e in errs))
pr = json.load(open(W / "predict.json"))["cases"]
print("predict cases", list(pr), "boxes", len(pr["case"]["boxes"]), "top scores", sorted(pr["case"]["scores"])[-3:])
PY'
```

Printout:

```
swept vs val_predictions: cases 2 all consistent; max abs diff 1.0
gt instances 10 max coordinate error 0.0 exact 10
predict cases ['case'] boxes 789 top scores [0.022872405126690865, 0.023091508075594902, 0.03112873062491417]
```

All three checks matched the expected shape: the swept-vs-nnDetection assertions held, gt max coordinate error was
0.0 (no resampling on this toy task), and predict returned exactly one case `case`.
