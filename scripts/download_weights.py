from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="zhixianhu/muxGel",
    repo_type="dataset",
    local_dir=".",
    allow_patterns="outputs/*"
)
print("✅ Checkpoints downloaded successfully!")