from torch import nn

class LinearClassifier(nn.Module):
    def __init__(self, d, c):
        super().__init__()
        self.net = nn.Linear(d, c)
    def forward(self, x):
        return self.net(x)

class MLPClassifier(nn.Module):
    def __init__(self, d, c):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, c))
    def forward(self, x):
        return self.net(x)


def make_model(kind, d, c):
    if kind == 'Linear': return LinearClassifier(d,c)
    if kind == 'MLP': return MLPClassifier(d,c)
    raise ValueError(f'Unknown model type: {kind}')
