# Simple PyTorch Example

A minimal example that trains a small neural network to approximate **y = 2x + 1** on synthetic data.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python simple_example.py
```

You should see the loss decrease and a final prediction at `x=1` close to `3`.

## What it does

- Builds synthetic data with `y = 2x + 1` plus noise.
- Defines a tiny MLP: input → 16 units (ReLU) → output.
- Trains with MSE loss and Adam for 200 epochs.
- Prints loss every 50 epochs and a sample prediction at the end.
