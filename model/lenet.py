import torch
import pathlib

class LeNetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lenet_stack = torch.nn.Sequential(
            torch.nn.Conv2d(1, 12, kernel_size=3, padding=2, stride=1), # output shape: 12 x 28 x 28
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2, stride=2), # output shape: 12 x 14 x 14

            # output shape: 32 x 12 x 12
            torch.nn.Conv2d(12, 32, kernel_size=3, padding=0, stride=1),
            torch.nn.ReLU(),
            # output shape: 32 x 6 x 6
            torch.nn.AvgPool2d(2, stride=2),

            torch.nn.Flatten(), # output shape: 32*6*6 = 1152

            # Fully connected layers
            torch.nn.Linear(32*6*6, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.Sigmoid(),
            torch.nn.Linear(84, 10)
        )

    def forward(self, x):
        logits = self.lenet_stack(x)
        return logits


torch_filename = str(pathlib.Path(__file__).parent / "lenet.pth")
onnx_filename = str(pathlib.Path(__file__).parent / "lenet.onnx")
sample_filename = str(pathlib.Path(__file__).parent / "lenet_sample_tensors.pt")
seed = 42
