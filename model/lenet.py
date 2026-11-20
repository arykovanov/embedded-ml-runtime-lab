import numpy as np
import pathlib
import struct
import torch
from urllib.request import Request, urlopen

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
tflight_filename = str(pathlib.Path(__file__).parent / "lenet.tflite")
tflight_quantized_filename = str(pathlib.Path(__file__).parent / "lenet_quantized.tflite")

onnx_filename = str(pathlib.Path(__file__).parent / "lenet.onnx")
sample_filename = str(pathlib.Path(__file__).parent / "sample_images.bin")
labels_filename = str(pathlib.Path(__file__).parent / "sample_labels.bin")
cpp_logits_filename = pathlib.Path(__file__).parent / "cpp_logits.bin"

seed = 42

def save_tensor(images, filename):
    with open(filename, "wb") as f:
        f.write(struct.pack("<I", images.ndim))
        f.write(struct.pack(f"<{images.ndim}q", *images.shape))
        b = images.tobytes(order="C")
        f.write(struct.pack("<I", len(b)))
        f.write(b)

def load_tensor(filename, dtype="<f4"):
    dtype = np.dtype(dtype)
    with open(filename, "rb") as f:
        def read_exact(size):
            data = f.read(size)
            if len(data) != size:
                raise ValueError("Truncated tensor file")
            return data

        rank, = struct.unpack("<i", read_exact(4))
        if rank < 0 or rank > 32:
            raise ValueError(f"Invalid tensor rank: {rank}")

        shape = struct.unpack(f"<{rank}q", read_exact(rank * 8))
        if any(dim < 0 for dim in shape):
            raise ValueError(f"Invalid tensor shape: {shape}")

        print(f'Tensor shape = {shape}')
        data_size, = struct.unpack("<i", read_exact(4))
        element_count = 1
        for dim in shape:
            element_count *= dim
        if data_size != element_count * dtype.itemsize:
            raise ValueError(f"Tensor byte count {data_size} does not match shape {element_count}*{dtype.itemsize} and dtype {dtype}")

        data = read_exact(data_size)
        if f.read(1):
            raise ValueError("Unexpected trailing data in tensor file")

    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def esp32_inference(image, url, timeout=10.0):
    """Send one preprocessed 28x28 image to the ESP32 and return 10 float32 logits.

    image may be a NumPy array or a PyTorch tensor with 784 elements.
    url is the complete endpoint URL, for example http://192.168.1.123/.
    HTTP and connection errors propagate to the caller; invalid data raises ValueError.
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().to(device="cpu", dtype=torch.float32).numpy()
    pixels = np.asarray(image, dtype=np.float32).reshape(-1)
    if pixels.size != 28 * 28:
        raise ValueError(f"Expected 784 image elements, got {pixels.size}")
    if not np.isfinite(pixels).all():
        raise ValueError("Image values must be finite")

    body = (str(pixels.size) + " " + " ".join(format(float(pixel), ".9g") for pixel in pixels)).encode("ascii")
    request = Request(url, data=body, headers={"Content-Type": "text/plain; charset=ascii"}, method="POST")
    with urlopen(request, timeout=timeout) as response:
        fields = response.read().decode("ascii").split()

    if not fields:
        raise ValueError("Empty inference response from ESP32")
    count = int(fields[0])
    if count != 10 or len(fields) != count + 1:
        raise ValueError(f"Expected 10 logits, got count {count} and {len(fields) - 1} values")

    logits = np.asarray(fields[1:], dtype=np.float32)
    if not np.isfinite(logits).all():
        raise ValueError("ESP32 returned non-finite logits")
    return logits
