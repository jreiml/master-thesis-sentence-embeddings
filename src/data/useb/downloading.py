import os
import sys
import zipfile

import requests
from tqdm.auto import tqdm


def http_get(url, path):
    """
    Downloads a URL to a given path on disc
    """
    if os.path.dirname(path) != '':
        os.makedirs(os.path.dirname(path), exist_ok=True)

    req = requests.get(url, stream=True)
    if req.status_code != 200:
        print("Exception when trying to download {}. Response {}".format(url, req.status_code), file=sys.stderr)
        req.raise_for_status()
        return

    download_filepath = path+"_part"
    with open(download_filepath, "wb") as file_binary:
        content_length = req.headers.get('Content-Length')
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                progress.update(len(chunk))
                file_binary.write(chunk)

    os.rename(download_filepath, path)
    progress.close()


def unzip(zip_file, out_dir='./datasets/usep/'):
    if not os.path.isdir(zip_file.replace(".zip", "")):
        zip_ = zipfile.ZipFile(zip_file, "r")
        zip_.extractall(path=out_dir)
        zip_.close()    


def download(data_type):
    assert data_type in ['train', 'eval', 'all'], "Invalid mode!"

    if data_type in ['train', 'all']:
        url = 'https://public.ukp.informatik.tu-darmstadt.de/kwang/unsupse-benchmark/tsdae-evaluation/data-train.zip'
        file_name = "./datasets/usep/" + url.split('/')[-1]
        if not os.path.exists(file_name):
            http_get(url, file_name)
        unzip(file_name)

    if data_type in ['eval', 'all']:
        url = 'https://public.ukp.informatik.tu-darmstadt.de/kwang/unsupse-benchmark/tsdae-evaluation/data-eval.zip'
        file_name = "./datasets/usep/" + url.split('/')[-1]
        if not os.path.exists(file_name):
            http_get(url, file_name)
        unzip(file_name)


if __name__ == '__main__':
    download("all")
