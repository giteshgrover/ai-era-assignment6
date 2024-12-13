import torch
import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        
        dropout_rate = 0.2 # Dropout .2..Isn't it too high?
        # nn.Conv2d(inputChannel, NumOfKernels or Output Channels, kernel_size=3, padding=1)
        # nn.BatchNorm2d(inputChannel),

        # Convolution Layers ()
        self.features = nn.Sequential(
            # Convolution Layer 1
            nn.Conv2d(1, 6, kernel_size=3, padding=1), # Input: 1, Output: 6, RF: 3 (ImageSize 28x28)
            nn.BatchNorm2d(6), # Input: 1
            nn.Dropout(dropout_rate), 
            nn.ReLU(), # Activation Func
            nn.MaxPool2d(2),

            # Convolution Layer 2
            nn.Conv2d(6, 12, kernel_size=3, padding=1), # Input: 6, Output: 12, RF: 5
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_rate),
            nn.ReLU(),

            # Transit Layer
            nn.MaxPool2d(2), # Input: 12, Output: 12, RF: 6 ((ImageSize 14 x 14) Max pool reduces the image size by half, not the output channels)
          
            # # Convolution Layer 3
            # nn.Conv2d(12, 24, kernel_size=3, padding=1), # Input: 12, Output: 24, RF: 10
            # nn.BatchNorm2d(24),
            # nn.Dropout(dropout_rate),
            # nn.ReLU(),

            # # Transit Layer
            # nn.MaxPool2d(2), # Input: 24, Output: 24, RF: 11 (ImageSize 7 x 7)
            # nn.Conv2d(24, 6, kernel_size=1, padding=1), # 1x1 Mixer Input: 24, Output: 6, RF: 11


            # # Convolution Layer 4
            # nn.Conv2d(6, 12, kernel_size=3, padding=1), # Input: 6, Output: 12, RF: 19
            # nn.ReLU(),

        )

        
        self.classifier = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(12 * 7 * 7, 32),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(32, 10)
        )
        
    def forward(self, x):
        x = self.features(x)
        x.size(0)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x 