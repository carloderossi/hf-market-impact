"""
submit_train_job.py
-------------------
Azure ML SDK v2 — Training Job Submitter.

  1. Build and submit a CommandJob 
  2. Stream / poll job status with live progress reporting

"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import Model
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

from auth import getMLClient

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_NAME = "hf-market-impact-exp"
ENVIRONMENT_NAME = "hf-market-impact-train" #"credit-risk-train@latest"

# Polling interval (seconds) when streaming is not available
POLL_INTERVAL = 10

from pathlib import Path

def get_train_env_path() -> Path:
    # File location: {proj_root}/src/train/your_script.py
    current_file = Path(__file__).resolve()
    proj_root = current_file.parents[1]   # go from src → proj_root
    return proj_root / "src" / "env.yml"

def create_env(ml_client):
    env_name = ENVIRONMENT_NAME
    env_version = "v1"

    try:
        path = get_train_env_path()
        print(path, path.exists())

        env = ml_client.environments.get(env_name, env_version)
        print(f"Environment already exists: {env.name}:{env.version}")
    except Exception:
        print("Creating new environment...")
        env = Environment(
            name=env_name,
            # version=env_version, # yaml file hashing comparison
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
            conda_file=get_train_env_path() 
        )
        ml_client.environments.create_or_update(env)
        print(f"Environment created: {env.name} - {env.version}")

    return env

# ---------------------------------------------------------------------------
# Job builder
# ---------------------------------------------------------------------------

def build_job(env: Environment) -> command:
    """Construct the Azure ML CommandJob object."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = f"hf-market-impact-train-{timestamp}"
    display_name = (
        f"High Frequency Marcket Impact ({timestamp})"
    )

    # Build the CLI command string — mirrors what was in the YAML
    command_str = (
        f"python ./hf-market-impact/run_models.py"
        f" --output_path ./outputs"
    )

    job = command(
        name=job_name,
        display_name=display_name,
        description=(
            f"DoubleML vs CausalForest "
            f"Submitted: {timestamp}."
        ),
        experiment_name=EXPERIMENT_NAME,
        tags={
            "project":      "hf-market-impact",
            "author":       "carlo",
            "submitted_at": timestamp,
        },
        # ── Code: repo root (src/ must be importable) ──────────────────────
        code="./src", #str(Path(__file__).parent.parent),   # project root
        command=command_str,
        environment=env, #f"azureml:{ENVIRONMENT_NAME}",
        compute=f"ml-ai300cdr-cluster", #f"azureml:{args.compute_cluster}",

        # ── Resources ───────────────────────────────────────────────────────
        instance_count=1,
    )

    return job


# ---------------------------------------------------------------------------
# Job submission & polling
# ---------------------------------------------------------------------------

TERMINAL_STATES = {"Completed", "Failed", "Canceled", "NotResponding"}


def submit_and_wait(ml_client: MLClient, job) -> tuple[str, str]:
    """
    Submit a job and block until it reaches a terminal state.
    Returns (job_name, final_status).
    """
    log.info("Submitting job: %s", job.name)
    submitted = ml_client.jobs.create_or_update(job)
    job_name = submitted.name
    print("Submitted job:", job_name)

    studio_url = getattr(submitted, "studio_url", None)
    if studio_url:
        log.info("Studio URL: %s", studio_url)

    log.info("Polling every %ds — waiting for terminal state...", POLL_INTERVAL)
    last_status = None

    while True:
        time.sleep(POLL_INTERVAL)
        current = ml_client.jobs.get(job_name)
        status = current.status

        if status != last_status:
            # New line + full status
            print(f"\nJob status: {status}", end="", flush=True)
            last_status = status
        else:
            # Same status → just print a dot
            print(".", end="", flush=True)

        if status in TERMINAL_STATES:
            print()  # final newline
            break

    log.info("Job finished with status: %s", last_status)
    return job_name, last_status


# ---------------------------------------------------------------------------
# Model registration
# ---------------------------------------------------------------------------

def read_eval_metrics(ml_client: MLClient, job_name: str) -> dict:
    """
    Attempt to read eval_metrics.json from the job output.
    Falls back gracefully — registration proceeds even without metrics.
    """
    try:
        # Download the model output artifact to a temp location
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ml_client.jobs.download(
                name=job_name,
                output_name="model_output",
                download_path=tmp,
            )
            metrics_path = Path(tmp) / "named-outputs" / "model_output" / "eval_metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                log.info("Eval metrics loaded: %s", json.dumps(metrics))
                return metrics
    except Exception as exc:
        log.warning("Could not read eval_metrics.json: %s", exc)
    return {}


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(job_name: str, job_status: str):
    width = 80
    bar = "─" * width
    print(f"\n┌{bar}┐")
    print(f"│{'  High Frequency Marcket Impact — Job Summary':^{width}}│")
    print(f"├{bar}┤")
    rows = [
        ("Job name",       job_name),
        ("Final status",   job_status),
        ("Experiment",     EXPERIMENT_NAME),
    ]
    for label, value in rows:
        print(f"│  {label:<22}{str(value):<{width - 24}}│")
    print(f"└{bar}┘\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit Azure ML Credit Risk training job and register the model."
    )
    # Workspace identity
    parser.add_argument("--compute_cluster", default="ml-ai300cdr-cluster",
                        help="Name of the AML compute cluster (without 'azureml:' prefix)")

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Connect ─────────────────────────────────────────────────────────────
    log.info("Connecting to workspace 'mlw-ai300-cdrlabs' ...")
    ml_client = getMLClient(None)

    # ── Environment ────────────────────────────────────────────────────────────
    env = create_env(ml_client)

    # ── Build job ────────────────────────────────────────────────────────────
    job = build_job(env)

    print(f"Validating compute '{job.compute}'")
    ml_client.compute.get(job.compute)

    if True:
        log.info("JOB configuration:")
        log.info("  name:        %s", job.name)
        log.info("  display:     %s", job.display_name)
        log.info("  experiment:  %s", job.experiment_name)
        log.info("  environment: %s", job.environment)
        log.info("  compute:     %s", job.compute)
        log.info("  command:     %s", job.command)
        log.info("  inputs:      %s", {k: v.path for k, v in job.inputs.items()})
        log.info("  tags:        %s", job.tags)

    # ── Submit + poll ────────────────────────────────────────────────────────
    job_name, job_status = submit_and_wait(ml_client, job)

    if job_status != "Completed":
        log.error("Job did not complete successfully (status=%s).", job_status)
        print_summary(job_name, job_status, None, args)
        sys.exit(1)

    # ── Read metrics from job artifact ───────────────────────────────────────
    # metrics = read_eval_metrics(ml_client, job_name)

    # ── Summary ──────────────────────────────────────────────────────────────
    print_summary(job_name, job_status)


if __name__ == "__main__":
    main()
