"""Console entry point powered by Click."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from .pipeline import run_pipeline


@click.command()
@click.argument(
    "input_dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False, resolve_path=True),
)
@click.argument("repo_id")
@click.option(
    "--work-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Directory for intermediate files (defaults to ./output).",
)
@click.option("--model-id", default="openai/whisper-base", show_default=True)
@click.option("--device", default=None, metavar="DEVICE", help="Force compute device (cuda, mps, cpu).")
@click.option(
    "--hf-token",
    default=None,
    envvar="HUGGINGFACEHUB_API_TOKEN",
    help="Hugging Face token (uses HUGGINGFACEHUB_API_TOKEN if unset).",
)
@click.option(
    "--speaker-prefix",
    default="Speaker 0: ",
    show_default=True,
    help="Prefix prepended to each transcript.",
)
def main(
    input_dir: Path,
    repo_id: str,
    work_dir: Optional[Path],
    model_id: str,
    device: Optional[str],
    hf_token: Optional[str],
    speaker_prefix: str,
) -> None:
    """Segment speech, run Whisper ASR, and push a dataset to the HF Hub."""
    resolved_work_dir = (work_dir or Path.cwd() / "output").expanduser().resolve()
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.exists():  # Defensive: Click already checks, but keep for clarity
        raise click.UsageError(f"Input directory not found: {input_dir}")
    dataset = run_pipeline(
        input_dir=input_dir,
        repo_id=repo_id,
        work_dir=resolved_work_dir,
        model_id=model_id,
        device=device,
        token=hf_token,
        speaker_prefix=speaker_prefix,
    )
    click.echo(f"Uploaded {len(dataset)} records to {repo_id}")
