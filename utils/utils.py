import hashlib
import os
import shutil


def get_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def clear_dir(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        os.remove(file_path)


if __name__ == '__main__':
    sha256 = get_sha256('/temp/download_upload/edf_input\\002914.edf')
    print(sha256)
