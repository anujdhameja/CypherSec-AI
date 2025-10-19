import torch
from ..utils.objects import stats


def softmax_accuracy(probs, all_labels):
    acc = (torch.argmax(probs) == all_labels).sum()
    acc = torch.div(acc, len(all_labels) + 0.0)
    return acc


class Step:
    # Performs a step on the loader and returns the result
    def __init__(self, model, loss_function, optimizer):
        self.model = model
        self.criterion = loss_function
        self.optimizer = optimizer

    def __call__(self, i, x, y):
        # Clear gradients first (moved to beginning for stability)
        if self.model.training:
            self.optimizer.zero_grad()
        
        out = self.model(x)
        target = y.squeeze().long()  # Ensure correct target format
        loss = self.criterion(out, target)
        
        # Calculate accuracy with proper target format
        pred = out.argmax(dim=1)
        acc = (pred == target).float().mean()

        if self.model.training:
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"⚠️ NaN loss detected at batch {i}, skipping")
                return stats.Stat(out.tolist(), 0.0, 0.0, y.tolist())
            
            # Backward pass
            loss.backward()
            
            # CRITICAL: Gradient clipping for stability
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Check for exploding gradients
            if grad_norm > 10.0:
                print(f"⚠️ Large gradient norm: {grad_norm:.2f} at batch {i}")
            
            # Parameter update
            self.optimizer.step()

        # print(f"\tBatch: {i}; Loss: {round(loss.item(), 4)}", end="")
        return stats.Stat(out.tolist(), loss.item(), acc.item(), y.tolist())

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
