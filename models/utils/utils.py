import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self, weight) -> None:
        super().__init__()
        self.weight = weight

    def forward(self):
        return self.weight

def dynm_fun(f):
    """A wrapper for the dynamical function"""

    def wrapper(self, t, x):
        new_fun = lambda t, x: f(self, t, x)
        return new_fun(t, x)

    return wrapper

def check_accuracy(model, loader, device):
    """
    Returns the fraction of correct predictions for a given model on a given dataset.
    """
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for x, y_true in loader:
            x = x.to(device)
            y_true = y_true.to(device)
            y_pred = model(x)
            correct += (y_pred.argmax(1) == y_true).type(torch.float).sum().item()
    return correct / len(loader.dataset)


def get_activation(activation, name):
    def hook(model, input, output):
        activation[name] = output
    return hook

def process_activations(activation, not_keys):
    """
    Detatches the activations of the layers which are not in not_keys and
    converts them into the shape (batch_size, number of neurons).
    """
    for key, item in activation.items():
        if key not in not_keys:
            activation[key] = activation[key].detach().cpu()
            activation[key] = (
                activation[key].view(activation[key].size()[0], -1).numpy()
            )
    return activation
