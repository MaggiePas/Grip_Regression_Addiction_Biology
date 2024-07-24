
import torch.nn as nn
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, RegressorMixin

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


class SKLearnWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


def create_traditional_model(model_type, **kwargs):
    if model_type == 'svr':
        return SKLearnWrapper(SVR(**kwargs))
    elif model_type == 'ridge':
        return SKLearnWrapper(Ridge(**kwargs))
    else:
        raise ValueError(f"Unknown model type: {model_type}")