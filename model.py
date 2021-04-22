import torch

class AudioClassifier (torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.stft_features = torch.nn.Sequential(
            torch.nn.BatchNorm2d(1),
            torch.nn.Conv2d(1, 16, kernel_size=(7, 3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, kernel_size=(7, 3)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(7,3)),
            torch.nn.Dropout2d(0.1),
            torch.nn.Conv2d(16, 32, kernel_size=3),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 32, kernel_size=3),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=(3,3)),
            torch.nn.Dropout2d(0.1),
            torch.nn.Conv2d(32, 128, kernel_size=3),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(44,20)),
            torch.nn.Dropout2d(0.1)
        )

        self.cqt_features = torch.nn.Sequential(
            torch.nn.BatchNorm2d(1),
            torch.nn.Conv2d(1, 16, kernel_size=(7, 3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, kernel_size=(7, 3)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(7,3)),
            torch.nn.Dropout2d(0.1),
            torch.nn.Conv2d(16, 32, kernel_size=3),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 32, kernel_size=3),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=(3,3)),
            torch.nn.Dropout2d(0.1),
            torch.nn.Conv2d(32, 128, kernel_size=3),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(1,20)),
            torch.nn.Dropout2d(0.1)
        )

        self.final_features = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 41),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        #Split input
        x1, x2 = x
        #add channel columns
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        #blocks for stft spectrogram
        x1 = self.stft_features(x1) 
        #blocks for cqt spectrogram
        x2 = self.cqt_features(x2) 
        #join the outputs
        x = torch.cat((x1, x2), 1)
        x = x.squeeze()
        #Output
        x = self.final_features(x)
        return x