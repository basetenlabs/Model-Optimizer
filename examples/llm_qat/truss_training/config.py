from truss_train import definitions
from truss.base import truss_config

project_name = "qwen3-235b-qat"

BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel"

training_runtime = definitions.Runtime(
    start_commands=[
        "/bin/sh -c 'chmod +x ./run.sh && ./run.sh'"
    ],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(name="parsed_hf_token"),
        "DATASET_HF_TOKEN": definitions.SecretReference(name="harrypartridge_hf_token_whetstone"),
        "GH_TOKEN": definitions.SecretReference(name="GITHUB_TOKEN"),
    },
    cache_config=definitions.CacheConfig(
        enabled=True,
    ),
    checkpointing_config=definitions.CheckpointingConfig(
        enabled=True,
    ),
)

training_compute = definitions.Compute(
    node_count=1,
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.B200,
        count=8,
    ),
)

my_training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

first_project_with_job = definitions.TrainingProject(
    name=project_name, job=my_training_job
)
