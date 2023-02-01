import os
import argparse
import urllib.request

URL_1 = "https://dados.agricultura.gov.br/dataset/baefdc68-9bad-4204-83e8-f2888b79ab48/resource/97f29a77-4e7e-44bf-99b3-a2d75911b6bf/download/psrdadosabertos2006a2015csv.csv"
URL_2 = "https://dados.agricultura.gov.br/dataset/baefdc68-9bad-4204-83e8-f2888b79ab48/resource/54e04a6b-15b3-4bda-a330-b8e805deabe4/download/psrdadosabertos2016a2021csv.csv"
URL_3 = "https://dados.agricultura.gov.br/dataset/baefdc68-9bad-4204-83e8-f2888b79ab48/resource/8fd05876-7679-44c6-ae9a-50e07c765a25/download/psrdadosabertos2022csv.csv"


def download_file(url, local_path):
    # Open the url image, set stream to True, this will return the stream content.
    with urllib.request.urlopen(url) as url:
        s = url.read()

    # Open a local file with wb ( write binary ) permission.
    with open(local_path, "wb") as f:
        f.write(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and persist the data from URL onto a local path."
    )
    parser.add_argument("local_path", help="The local path to persist the data")
    args = parser.parse_args()
    local_path = args.local_path

    if not os.path.exists(local_path):
        os.makedirs(local_path)

    file_name = URL_1.split("/")[-1]
    download_file(URL_1, local_path + file_name)
    print(f"File saved at {local_path}")

    file_name = URL_2.split("/")[-1]
    download_file(URL_2, local_path + file_name)
    print(f"File saved at {local_path}")

    file_name = URL_3.split("/")[-1]
    download_file(URL_3, local_path + file_name)
    print(f"File saved at {local_path}")
