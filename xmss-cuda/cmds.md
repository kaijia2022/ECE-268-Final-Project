1. Build the Container Image

Bash
sudo docker build -t xmss-cuda .

2. Run the Container with GPU Passthrough
Use the CDI --device flag to inject the GPU into the container environment.

Bash
sudo docker run --rm --device nvidia.com/gpu=all xmss-cuda

sudo docker run --rm --gpus all xmss-cuda


enter container:
sudo docker run --rm -it --gpus all xmss-cuda bash