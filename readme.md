## FFMPEG
https://www.ffmpeg.org/download.html

## Troubleshooting

#### 'CUDA not available' error
If your GPU supports cuda but still getting this error, 
try installing the correct version of torch.
Check the compatibility of your GPU with the version of torch 
from [here](https://pytorch.org/get-started/previous-versions/).

#### Check Cuda version
```shell
nvcc --version
```

#### Example
```
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
