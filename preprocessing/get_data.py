import os
import argparse
import urllib.request
import logging
from pathlib import Path


# These are public data URLs which might change in the future
URL_1 = "https://dados.agricultura.gov.br/dataset/baefdc68-9bad-4204-83e8-f2888b79ab48/resource/97f29a77-4e7e-44bf-99b3-a2d75911b6bf/download/psrdadosabertos2006a2015csv.csv"
URL_2 = "https://dados.agricultura.gov.br/dataset/baefdc68-9bad-4204-83e8-f2888b79ab48/resource/54e04a6b-15b3-4bda-a330-b8e805deabe4/download/psrdadosabertos2016a2021csv.csv"
URL_3 = "https://dados.agricultura.gov.br/dataset/baefdc68-9bad-4204-83e8-f2888b79ab48/resource/8fd05876-7679-44c6-ae9a-50e07c765a25/download/psrdadosabertos2022csv.csv"
URLs = [URL_1, URL_2, URL_3]


def download_file(url, local_path):
    # Open the url image, set stream to True, this will return the stream content.
    with urllib.request.urlopen(url) as url:
        s = url.read()

    # Open a local file with wb ( write binary ) permission.
    with open(local_path, "wb") as f:
        f.write(s)


def get_file_by_urls(local_path, URLs: list):
    full_path = Path().parent.absolute() / Path(local_path)
    return [full_path / U.split("/")[-1] for U in URLs]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and persist the data from URL onto a local path."
    )
    parser.add_argument("local_path", help="The local path to persist the data")
    args = parser.parse_args()
    local_path = args.local_path
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    file_list = get_file_by_urls(local_path=local_path, URLs=URLs)
    if all([os.path.exists(f) for f in file_list]):
        print("Files already available")
    else:
        [download_file(u, local_path + u.split("/")[-1]) for u in URLs]
