# # Base image
# gcr.io/deeplearning-platform-release/pytorch-cu121.2-2.py310
# deeplearning-platform-release/gcr.io/pytorch-cu121.2-2.py310
# pytorch/pytorch:latest-cuda11.8-cudnn8-devel
FROM gcr.io/deeplearning-platform-release/pytorch-cu121.2-2.py310
WORKDIR /
# # Install dependencies
RUN pip install h5py numpy wandb
# Copy training script and other required directories/files
COPY train_transformer.py /train_transformer.py
COPY util/ /util/
COPY model/ /model/
COPY tests/ /tests/
# Create an empty checkpoint directory
RUN mkdir /ckpt
# # Set the entrypoint
ENTRYPOINT ["python","-m","train_transformer"]
