from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./src/model",
    repo_id="MJPT2/llm-planning-reasoning",
    commit_message="Upload fine-tuned model"
)
