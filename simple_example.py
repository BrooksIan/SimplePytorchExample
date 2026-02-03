"""
Simple PyTorch example: train a small neural network to fit y ≈ 2x + 1.
"""

import torch
import torch.nn as nn

# 1. Create synthetic data: y = 2x + 1 + small noise
torch.manual_seed(42)
x = torch.randn(100, 1) * 3  # 100 points in roughly [-9, 9]
y = 2 * x + 1 + 0.1 * torch.randn(100, 1)

# 2. Define a simple model (one hidden layer)
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)

model = SimpleNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 3. Train
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# 4. Quick check: predict for x = 1 (should be ≈ 3)
model.eval()
with torch.no_grad():
    test_x = torch.tensor([[1.0]])
    pred_y = model(test_x)
    print(f"\nAt x=1, predicted y ≈ {pred_y.item():.2f} (expected ≈ 3)")
