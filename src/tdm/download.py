"""
TODO Replace this entire module with pooch
"""

from tqdm.auto import tqdm
import requests
import zipfile


def unzip_file(zip_file_path, extract_to_path):
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to_path)


def download_file(url, filename):
    response = requests.get(url, stream=True)
    content_length = response.headers.get("content-length")
    if content_length is None:
        raise RuntimeError("Failed to download file")
    else:
        total = int(content_length)
        chunk_size = 1024

        with open(filename, "wb") as file:
            for chunk in tqdm(
                response.iter_content(chunk_size=chunk_size),
                total=int(total / chunk_size),
                unit="KB",
            ):
                file.write(chunk)
