from huggingface_hub import snapshot_download
from pathlib import Path
import subprocess, sys, os

PAIRS = [
    ("it","en","Helsinki-NLP/opus-mt-it-en"),
    ("en","it","Helsinki-NLP/opus-mt-en-it"),
]

BASE = Path(__file__).parent.resolve()
RAW_DIR = BASE / "raw_models"
CT2_DIR = BASE / "ct2_models"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CT2_DIR.mkdir(parents=True, exist_ok=True)

def run(cmd: list[str]):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

def ct2_converter_path() -> str:
    exe = "ct2-transformers-converter.exe" if os.name == "nt" else "ct2-transformers-converter"
    # use the venv Scripts folder (sibling of python.exe)
    return str(Path(sys.executable).with_name(exe))

def main():
    converter = ct2_converter_path()
    registry_lines = []
    for src, tgt, repo in PAIRS:
        print(f"=== {src}->{tgt} :: {repo}")
        local = Path(snapshot_download(repo_id=repo, cache_dir=str(RAW_DIR)))

        spm_src = local / "source.spm"
        spm_tgt = local / "target.spm"
        if not spm_src.exists() or not spm_tgt.exists():
            raise FileNotFoundError(f"Missing SentencePiece files in {local}")

        out_dir = CT2_DIR / f"{src}-{tgt}-int8"
        out_dir.mkdir(parents=True, exist_ok=True)

        run([
            converter,
            "--model", str(local),
            "--output_dir", str(out_dir),
            "--quantization", "int8",
            "--copy_files", "source.spm", "target.spm",
            "--force",
        ])

        registry_lines.append(f"{src}->{tgt}\t{out_dir}\t{spm_src}\t{spm_tgt}\t{local}")

    (BASE / "registry.tsv").write_text("\n".join(registry_lines), encoding="utf-8")
    print("Wrote registry.tsv")

if __name__ == "__main__":
    main()
