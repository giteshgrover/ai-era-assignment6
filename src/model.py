import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        
        dropout_rate = 0.1 # Dropout
        # nn.Conv2d(inputChannel, NumOfKernels or Output Channels, kernel_size=3, padding=1)
        # nn.BatchNorm2d(inputChannel),

        # Convolution Layers ()
        self.features = nn.Sequential(
            # Convolution Layer 1
            # (ImageSize 28x28)
            nn.Conv2d(1, 12, kernel_size=3, padding=1), # Input: 1, Output: 6, RF: 3 
            nn.BatchNorm2d(12), # Input: 1
            nn.Dropout(dropout_rate), 
            nn.ReLU(), # Activation Func

            # Convolution Layer 2
            nn.Conv2d(12, 24, kernel_size=3, padding=1), # Input: 6, Output: 12, RF: 5
            nn.BatchNorm2d(24),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            

            # Convolution Layer 3
            # nn.Conv2d(12, 24, kernel_size=3, padding=1), # Input: 12, Output: 24, RF: 7
            # nn.BatchNorm2d(24),
            # nn.Dropout(dropout_rate),
            # nn.ReLU(),

            # Transition Layer
            nn.MaxPool2d(2), # Input: 24, Output: 24, RF: 8 (Max pool reduces the image size by half, not the output channels)
            # ImageSize 14 x 14
            nn.Conv2d(24, 12, kernel_size=1, padding=0),

            # ImageSize 14 x 14
            # Convolution Layer 4
            nn.Conv2d(12, 16, kernel_size=3, padding=0), # Input: 6, Output: 12, RF: 12
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_rate),
            nn.ReLU(),

            # Convolution Layer 5
            nn.Conv2d(16, 32, kernel_size=3, padding=0), # Input: 12, Output: 24, RF: 16
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_rate),
            nn.ReLU(),

            # Convolution Layer 5
            nn.Conv2d(32, 32, kernel_size=3, padding=0), # Input: 12, Output: 24, RF: 16
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_rate),
            nn.ReLU(),

            # Transit Layer
            # nn.MaxPool2d(2), # Input: 24, Output: 24, RF: 17 (ImageSize 7 x 7)
            # nn.Conv2d(24, 6, kernel_size=1, padding=0), # 1x1 Mixer Input: 24, Output: 6

            # # ImageSize 7 x 7
            # # Convolution Layer 6
            # nn.Conv2d(6, 12, kernel_size=3, padding=1), # Input: 6, Output: 12, RF: 25
            # nn.BatchNorm2d(12),
            # nn.Dropout(dropout_rate),
            # nn.ReLU(),

            # # Convolution Layer 7
            # nn.Conv2d(12, 24, kernel_size=3, padding=1), # Input: 12, Output: 12, RF: 33
            # nn.BatchNorm2d(24),
            # nn.Dropout(dropout_rate),
            # nn.ReLU(),

         # Output Layer
            # ImageSize 7 x 7
            nn.Conv2d(32, 10, kernel_size=1, padding=0), # 1x1 Mixer Input: 24, Output: 10

            nn.AdaptiveAvgPool2d(1), # Input: 7 x 7 x 10 Output: 1 x 1 x 10
            # Dropoff, batch normalization and Relu are never done in the last layer
            # Image Size 1 x 1 (x 10)
            # nn.Conv2d(10, 10, kernel_size=1, padding=0),

        )

        
        # self.classifier = nn.Sequential(
        #     # nn.Dropout(0.5),
        #     nn.Linear(24 * 5 * 5, 12),
        #     nn.ReLU(),
        #     # nn.Dropout(0.5),
        #     nn.Linear(12, 10)
        # )
        
    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        x = x.view(-1, 10)
        x = F.log_softmax(x, dim=1)
        # print ("###############################")
        # print(x.shape)
        # print ("###############################")
        # print(x.shape)
        return x 
