# VLASH LIBERO Setup

This setup mirrors OpenVLA's LIBERO flow:

1. install the external `LIBERO` repo itself
2. install a small VLASH-side requirements file for extra simulation dependencies

## Recommended layout

The most reliable layout inside a Linux container is:

```bash
/workspace/
  LIBERO/
  vlash/
```

Keeping `LIBERO` and `vlash` as sibling folders makes editable installs easy to reason about and matches the example config paths in this repo.

## Important rule

After `conda activate vlash`, always use `python3 -m pip`, not bare `pip`.

That guarantees the package install goes into the same interpreter that later runs:

```bash
python3 -m vlash.eval.run_libero_eval ...
```

## Install order

From the VLASH environment:

```bash
conda activate vlash

cd /workspace/LIBERO
python3 -m pip uninstall -y libero
python3 -m pip install .

cd /workspace/vlash
python3 -m pip install -e .
python3 -m pip install -r examples/eval/libero_requirements.txt
```

If you have not cloned LIBERO yet:

```bash
cd /workspace
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
python3 -m pip install .
```

Use a regular install for LIBERO instead of `python3 -m pip install -e .`.
In practice, LIBERO's package layout can appear importable only when your current working
directory is `/workspace/LIBERO`, which makes editable installs look successful even when
`python3 -m vlash.eval.run_libero_eval ...` will still fail from another directory.

## Sanity checks

Verify that the current Python can see the required packages:

```bash
python3 -m pip show libero
cd /tmp
python3 -c "import sys; print(sys.executable)"
python3 -c "from libero.libero import benchmark; print('libero ok')"
python3 -c "import cv2, imageio, robosuite, bddl; print('deps ok')"
```

Run the `libero` import test from a neutral directory such as `/tmp`, not from `/workspace/LIBERO`.
If you test from `/workspace/LIBERO`, Python may import directly from the source tree and hide a broken install.

## Run eval

Inside a Linux container, use the Linux path to the config:

```bash
python3 -m vlash.eval.run_libero_eval --config /workspace/vlash/examples/eval/libero_sim.yaml
```

Do not pass a Windows path such as `C:/vlas/vlash/examples/eval/libero_sim.yaml` inside the container.
