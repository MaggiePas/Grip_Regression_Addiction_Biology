
import torch.nn as nn

class MLP(nn.Module):
    '''
    Multilayer Perceptron for regression.
    '''
    def __init__(self, input_size, hidden_size, output_size, p):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p),
            nn.Linear(hidden_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.LeakyReLU(),
            nn.Dropout(p),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.BatchNorm1d(hidden_size//4),
            nn.LeakyReLU(),
            nn.Dropout(p),
            nn.Linear(hidden_size//4, 1)
        )
    def forward(self, x):
        '''
          Forward pass
        '''
        return self.layers(x)