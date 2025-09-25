"""Core audio segmentation, transcription, and dataset utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
import torchaudio
from datasets import Audio, Dataset
from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
from tqdm.auto import tqdm
from transformers import pipeline as hf_pipeline


_SUPPORTED_EXTENSIONS = {".wav", ".mp3"}


def _iter_audio_files(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in _SUPPORTED_EXTENSIONS:
            yield path


def segment_audio(input_dir: Path, segments_dir: Path, vad_sampling_rate: int = 16_000) -> List[Path]:
    """Split incoming audio files into voice segments saved under ``segments_dir``."""
    segments_dir.mkdir(parents=True, exist_ok=True)
    model = load_silero_vad()
    created_segments: List[Path] = []

    for audio_path in tqdm(list(_iter_audio_files(input_dir)), desc="VAD", unit="file"):
        wav = read_audio(str(audio_path), sampling_rate=vad_sampling_rate)
        timestamps = get_speech_timestamps(wav, model, sampling_rate=vad_sampling_rate)
        if not timestamps:
            continue

        audio_tensor, sample_rate = torchaudio.load(str(audio_path))
        for idx, segment in enumerate(timestamps, start=1):
            start = int(segment["start"] * sample_rate / vad_sampling_rate)
            end = int(segment["end"] * sample_rate / vad_sampling_rate)
            if end <= start:
                continue
            segment_tensor = audio_tensor[:, start:end]
            output_path = segments_dir / f"{audio_path.stem}_{idx:03d}.wav"
            torchaudio.save(str(output_path), segment_tensor, sample_rate)
            created_segments.append(output_path)

    return created_segments


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def transcribe_audio(paths: Sequence[Path], model_id: str = "openai/whisper-base", device: str | None = None) -> Dict[Path, str]:
    """Run Whisper transcription for each audio path and return text per file."""
    if not paths:
        return {}
    resolved_device = device or _resolve_device()
    pipe = hf_pipeline("automatic-speech-recognition", model=model_id, device=resolved_device)
    transcripts: Dict[Path, str] = {}

    for path in tqdm(paths, desc="Transcribe", unit="file"):
        result = pipe(str(path))
        transcripts[path] = result["text"].strip()

    return transcripts


def build_dataset(transcripts: Dict[Path, str], speaker_prefix: str = "Speaker 0: ") -> Dataset:
    """Create a Hugging Face dataset where audio references local files."""
    if not transcripts:
        return Dataset.from_dict({"audio": [], "text": []})

    items = [
        {"audio": str(path), "text": f"{speaker_prefix}{text}".strip()}
        for path, text in sorted(transcripts.items(), key=lambda item: str(item[0]))
    ]
    dataset = Dataset.from_list(items)
    dataset = dataset.cast_column("audio", Audio())
    return dataset


def push_dataset(dataset: Dataset, repo_id: str, token: str | None = None) -> None:
    """Upload dataset to the specified Hugging Face Hub repo."""
    dataset.push_to_hub(repo_id, token=token)


def run_pipeline(
    input_dir: Path,
    repo_id: str,
    work_dir: Path,
    model_id: str = "openai/whisper-base",
    device: str | None = None,
    token: str | None = None,
    speaker_prefix: str = "Speaker 0: ",
) -> Dataset:
    """Execute segmentation, transcription, dataset creation, and upload."""
    work_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = work_dir / "segments"
    transcripts_dir = work_dir / "transcripts"
    segments = segment_audio(input_dir, segments_dir)
    transcripts = transcribe_audio(segments, model_id=model_id, device=device)

    if transcripts:
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        for path, text in transcripts.items():
            transcript_path = transcripts_dir / f"{path.stem}.txt"
            transcript_path.write_text(text + "\n", encoding="utf-8")

    dataset = build_dataset(transcripts, speaker_prefix=speaker_prefix)
    push_dataset(dataset, repo_id, token=token)
    return dataset
