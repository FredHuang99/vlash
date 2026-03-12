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

After `conda activate vlash`, always use `python -m pip`, not bare `pip`.

That guarantees the package install goes into the same interpreter that later runs:

```bash
python -m vlash.eval.run_libero_eval ...
```

## Install order

From the VLASH environment:

```bash
conda activate vlash

cd /workspace/LIBERO
python -m pip install -e .

cd /workspace/vlash
python -m pip install -e .
python -m pip install -r examples/eval/libero_requirements.txt
```

If you have not cloned LIBERO yet:

```bash
cd /workspace
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
python -m pip install -e .
```

## Sanity checks

Verify that the current Python can see the required packages:

```bash
python -m pip show libero
python -c "import sys; print(sys.executable)"
python -c "import libero, cv2, imageio; print(libero.__file__)"
```

## Run eval

Inside a Linux container, use the Linux path to the config:

```bash
python -m vlash.eval.run_libero_eval --config /workspace/vlash/examples/eval/libero_sim.yaml
```

Do not pass a Windows path such as `C:/vlas/vlash/examples/eval/libero_sim.yaml` inside the container.
