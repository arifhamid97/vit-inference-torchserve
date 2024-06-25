
# TorchServe Inference Project

## Overview
This project demonstrates how to use TorchServe for serving PyTorch models. TorchServe is an open-source tool that makes it easy to deploy and manage PyTorch models at scale. It provides a robust and flexible serving architecture optimized for production environments.

## Features
- **Model Serving**: Easily deploy PyTorch models for inference.
- **Scalability**: Serve models in a scalable and efficient manner.
- **Monitoring**: Track and monitor model performance and resource usage.
- **Custom Handlers**: Implement custom inference logic using custom handlers.

## Prerequisites
- Python 3.6+
- PyTorch
- TorchServe
- TorchModelArchiver

## Installation

### PyTorch
Install PyTorch following the instructions on the [official website](https://pytorch.org/get-started/locally/).

### TorchServe
Install TorchServe and TorchModelArchiver:

```sh
pip install torchserve torch-model-archiver
```

## Getting Started

### Step 1: Train Your Model
Train your PyTorch model or use an existing pre-trained model. Save the model in a `.pt` or `.pth` file.

### Step 2: Create a Model Archive
Create a model archive file (`.mar`) using the TorchModelArchiver. This file packages your model, code, and dependencies for serving.

```sh
torch-model-archiver --model-name <model-name> --version 1.0 --serialized-file <path-to-model-file> --handler <handler-file> --extra-files <additional-files> --export-path model_store
```

- `--model-name`: Name of the model.
- `--serialized-file`: Path to the model file.
- `--handler`: Path to the handler file.
- `--extra-files`: Any additional files required by the model.
- `--export-path`: Directory to store the `.mar` file.

### Step 3: Start TorchServe
Start the TorchServe service to serve your model.

```sh
torchserve --start --model-store model_store --models <model-name>=<model-name>.mar
```

### Step 4: Make Inference Requests
Once TorchServe is running, you can send inference requests to the model.

```sh
curl -X POST http://127.0.0.1:8080/predictions/<model-name> -T <input-file>
```

## Custom Handlers
TorchServe allows you to define custom handlers to implement custom preprocessing, inference, and postprocessing logic. Below is an example of a custom handler:

```python
from ts.torch_handler.base_handler import BaseHandler
import torch

class MyHandler(BaseHandler):
    def preprocess(self, data):
        # Custom preprocessing logic
        return torch.tensor(data)

    def inference(self, data):
        # Custom inference logic
        return self.model(data)

    def postprocess(self, data):
        # Custom postprocessing logic
        return data.numpy().tolist()
```

Save this handler to a file (e.g., `my_handler.py`) and reference it when creating the model archive.

## Monitoring and Logging
TorchServe provides a comprehensive monitoring and logging framework. You can configure logging and monitor metrics to keep track of the model's performance and resource usage.

## Conclusion
This project provides a foundation for serving PyTorch models using TorchServe. By following the steps outlined in this README, you can deploy your own models and customize the serving logic to fit your needs.

For more information, refer to the [TorchServe documentation](https://pytorch.org/serve/).
