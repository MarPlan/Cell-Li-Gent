import os
import subprocess

from google.cloud import aiplatform


def create_dataset():
    # create a new bucket in your project
    subprocess.run(f"gsutil mb -l {LOCATION} {BUCKET_URI}", shell=True)
    print(f"Bucket {BUCKET_URI} created")
    # Copy data in parallel
    subprocess.run(f"gsutil -m cp -r {LOCAL_DATA_DIR} {BUCKET_URI}", shell=True)
    print("Data uploaded")


def run_training():
    # Run the job
    api_key_cmd = subprocess.getoutput( "op read op://Personal/wandb_api/credential --no-newline")
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    job = aiplatform.CustomContainerTrainingJob(
        display_name=JOB_DISPLAY_NAME,
        container_uri=CONTAINER_URI,
        staging_bucket=BUCKET_URI,
    )

    job.run(
        replica_count=1,
        machine_type=MACHINE_TYPE,
        accelerator_type=GPU_TYPE,
        accelerator_count=1,
        args=[
            f"--data_file='gs://{PROJECT_ID}-bucket/battery_data.h5'"
            # f"--data_file=/gcs/{PROJECT_ID}-bucket/battery_data.h5",
            f"--out_dir='gs://{PROJECT_ID}-bucket/ckpt/'"
            # f"--out_dir=/gcs/{PROJECT_ID}-bucket/ckpt/",
            f"--wandb_api_key={api_key_cmd}",
            "--print_gpu=1",
            "--device='cuda'",
            "--compile=1",
            "--batch_size=32",
            "--max_iters=20",
            "--gradient_accumulation_steps=16",
            "--eval_interval=10",
            "--wandb_log=1",
        ],
    )

    # Delete training job created
    job.delete(sync=False)

    # Delete Cloud Storage objects that were created
    # delete_bucket = False
    # if delete_bucket:
    #     subprocess.run(f"gsutil -m rm -r {BUCKET_URI}")


if __name__ == "__main__":
    # Initializes the gcloud CLI and sets up your configuration, including selecting the project and region
    # subprocess.run("gcloud init", shell=True)
    # Authenticates your gcloud CLI with your Google Cloud account, allowing to use gcloud commands
    # subprocess.run("gcloud auth login", shell=True)
    # subprocess.run("gcloud services enable compute.googleapis.com", shell=True)
    # subprocess.run("gcloud services enable artifactregistry.googleapis.com", shell=True)
    # subprocess.run("gcloud services enable aiplatform.googleapis.com", shell=True)
    # gcloud auth application-default login

    PROJECT_ID = subprocess.getoutput(
        "gcloud config list --format 'value(core.project)'"
    )
    LOCATION = "europe-west4"  # @param {type:"string"}
    BUCKET_URI = f"gs://{PROJECT_ID}-bucket"  # @param {type:"string"}
    LOCAL_DATA_DIR = os.path.abspath("data/train/battery_data.h5")
    # create_dataset()

    JOB_DISPLAY_NAME = "transformer-training"
    GPU_TYPE = "NVIDIA_TESLA_V100" # "NVIDIA_A100_80GB"
    MACHINE_TYPE = "n1-standard-8"  # a2-ultragpu-1g
    REPO_NAME = "transformer-app"  # create repo in artifacts registry
    CONTAINER_URI = (
        f"{LOCATION}-docker.pkg.dev/{PROJECT_ID}/{REPO_NAME}/transformer_image:latest"
    )
    # subprocess.run(
    #     f"gcloud artifacts repositories create {REPO_NAME} --repository-format=docker --location={LOCATION} --description='Docker repository'",
    #     shell=True,
    # )
    # subprocess.run(
    #     f"gcloud auth configure-docker {LOCATION}-docker.pkg.dev", shell=True
    # )
    # subprocess.run(f"docker build ./ -t {CONTAINER_URI}", shell=True)
    # subprocess.run(f"docker push {CONTAINER_URI}", shell=True)
    run_training()
    print("--------------SUCCESSFULLY FINISHED--------------")
