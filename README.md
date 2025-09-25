# VibeVoice Dataset Preparer

A tool to prepare datasets for training VibeVoice.

Docs coming soon.

For now:
```
uv pip install git+https://github.com/vibevoice-community/databuilder
databuilder input/ username/huggingface_dataset_name
```

`input/` is a folder that should have raw MP3/WAV files. Any length.

Silero VAD -> Whisper Transcription -> Hugging Face Dataset