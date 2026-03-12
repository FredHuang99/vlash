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
touch libero/__init__.py
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

Create `/workspace/LIBERO/libero/__init__.py` before installing.
The upstream LIBERO repo has `libero/libero/__init__.py` but not `libero/__init__.py`, while
its `setup.py` uses `find_packages()`. With modern packaging tooling, that can leave you in a
broken state where `python3 -m pip show libero` succeeds but `import libero` only works when
your current directory is `/workspace/LIBERO`.

Use a regular install for LIBERO instead of `python3 -m pip install -e .`.
That avoids relying on editable-install behavior for this legacy package layout.

## Sanity checks

Verify that the current Python can see the required packages:

```bash
python3 -m pip show libero
cd /tmp
python3 -c "import sys; print(sys.executable)"
python3 -c "from libero.libero import benchmark; print('libero ok')"
python3 -c "import cv2, imageio, robosuite, bddl, matplotlib; print('deps ok')"
```

Run the `libero` import test from a neutral directory such as `/tmp`, not from `/workspace/LIBERO`.
If you test from `/workspace/LIBERO`, Python may import directly from the source tree and hide a broken install.

## robosuite macro warnings

You may see warnings like:

```text
[robosuite WARNING] No private macro file found!
[robosuite WARNING] It is recommended to use a private macro file
```

These warnings are usually non-fatal for LIBERO evaluation.
They mean `robosuite` did not find its optional local `macros_private.py` override file.

If evaluation is otherwise running, you can ignore them.
If you want to silence the warnings and generate the optional file once, run:

```bash
python3 /root/miniconda3/envs/vlash/lib/python3.10/site-packages/robosuite/scripts/setup_macros.py
```

## matplotlib and NumPy

LIBERO's upstream `requirements.txt` pins old versions such as `numpy==1.22.4` and
`matplotlib==3.5.3`, and LIBERO imports `matplotlib.cm` from
`libero/libero/envs/env_wrapper.py` during environment setup.

However, VLASH depends on `lerobot==0.4.1`, and LeRobot currently requires:

- `packaging>=24.2,<26.0`
- `opencv-python-headless>=4.9.0,<4.13.0`
- `rerun-sdk>=0.24.0,<0.27.0`

In practice, those LeRobot-era packages are much happier in a NumPy 2 environment than in a
legacy NumPy 1 environment. If you downgrade NumPy to 1.x just to satisfy old LIBERO pins, you
can trigger conflicts with `opencv-python-headless` and `rerun-sdk`.

For the shared VLASH environment, the recommended approach is:

1. keep NumPy 2
2. install a NumPy-2-compatible `matplotlib`
3. keep `packaging` below 26 to satisfy LeRobot
4. use exactly one OpenCV wheel in the environment

If your environment was already mutated by older pins, repair it with:

```bash
python3 -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
python3 -m pip install --force-reinstall --no-cache-dir "packaging==25.0" "numpy>=2,<2.3" "matplotlib>=3.10.3,<4.0.0" "opencv-python>=4.9.0,<5.0.0"
```

Why not use LIBERO's original `matplotlib==3.5.3` here?

- that version was built for the old NumPy 1.x stack
- with NumPy 2.x it can fail with `_ARRAY_API not found` or `numpy.core.multiarray failed to import`
- newer Matplotlib releases are compatible with the LeRobot / NumPy 2 stack used by VLASH

## OpenCV package choice

`robosuite==1.4.1` and `reachy2-sdk` declare a dependency on `opencv-python`, while
LeRobot declares `opencv-python-headless`.

The OpenCV wheel maintainers explicitly recommend installing only one of these packages in a
single environment because they all provide the same `cv2` namespace.

That means there is no perfectly clean one-env solution here: pip will complain about one side
or the other. For LIBERO evaluation, prefer `opencv-python`, because that satisfies the direct
requirements of `robosuite` and `reachy2-sdk`.

If you want the most stable setup, use a separate conda env for LIBERO evaluation, for example:

```bash
conda create -n vlash-libero python=3.10
conda activate vlash-libero
conda install ffmpeg=7.1.1 -c conda-forge
cd /workspace/vlash
python3 -m pip install -e .
cd /workspace/LIBERO
touch libero/__init__.py
python3 -m pip install .
cd /workspace/vlash
python3 -m pip install -r /workspace/vlash/examples/eval/libero_requirements.txt
```

If you must keep a single shared env, install only `opencv-python` and accept that `pip` may
still warn that LeRobot asked for the headless wheel.

## Run eval

Inside a Linux container, use the Linux path to the config:

```bash
python3 -m vlash.eval.run_libero_eval --config /workspace/vlash/examples/eval/libero_sim.yaml
```

Do not pass a Windows path such as `C:/vlas/vlash/examples/eval/libero_sim.yaml` inside the container.
