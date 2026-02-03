# Simple PyTorch Example

A minimal example that trains a small neural network to approximate **y = 2x + 1** on synthetic data.

**Python:** 3.11 recommended.

## Setup

### Local / general

```bash
pip install -r requirements.txt
```

### Cloudera AI Workbench (Python 3.11)

Installing the default PyTorch (CUDA) in a notebook session can crash the kernel due to memory and in-session loading. Use the **CPU-only** build and install from a **terminal**, then **restart the session**:

1. **In a terminal** (not in a notebook cell), from the project directory:
   ```bash
   pip install --no-cache-dir -r requirements-workbench.txt
   ```
   Or install directly from the CPU index:
   ```bash
   pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Restart the session** (or start a new one). Do not import `torch` in the same session where you ran `pip install`.

3. Run your code or open a new notebook and run the example.

Using the CPU-only index avoids large CUDA dependencies and reduces memory during install; your code runs unchanged on CPU.

## Run

**Script:**
```bash
python simple_example.py
```

**Notebook:** Open `simple_example.ipynb` for a step-by-step version with install instructions (including Cloudera AI Workbench). Run cells in order; on Workbench, install PyTorch from a terminal and restart the session before running the notebook.

You should see the loss decrease and a final prediction at `x=1` close to `3`.

## What it does

- Builds synthetic data with `y = 2x + 1` plus noise.
- Defines a tiny MLP: input → 16 units (ReLU) → output.
- Trains with MSE loss and Adam for 200 epochs.
- Prints loss every 50 epochs and a sample prediction at the end.
