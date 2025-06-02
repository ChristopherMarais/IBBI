from huggingface_hub import hf_hub_download


def download_from_hf_hub(repo_id: str, filename: str) -> str:
    """
    Downloads a model file from a Hugging Face Hub repository.

    Args:
        repo_id (str): The ID of the repository (e.g., "your-username/my-model").
        filename (str): The name of the file to download from the repo.

    Returns:
        str: The local path to the downloaded file.
    """
    print(f"Downloading {filename} from Hugging Face hub repository '{repo_id}'...")
    local_model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    print("Download complete. Model cached at:", local_model_path)
    return local_model_path
