from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = ROOT / "README.md"

setup(
    name="databuilder",
    version="0.1.0",
    description="Lightweight speech dataset builder with Whisper and Silero VAD",
    long_description=README.read_text(encoding="utf-8") if README.exists() else "",
    long_description_content_type="text/markdown",
    author="",
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "transformers",
        "tqdm",
        "torchaudio",
        "torch",
        "soundfile",
        "librosa",
        "nemo_toolkit[asr]",
        "datasets",
        "silero-vad",
        "click",
        "huggingface_hub",
    ],
    entry_points={
        "console_scripts": ["databuilder=databuilder.cli:main"],
    },
)
