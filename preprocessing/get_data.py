import os
import argparse
import urllib.request

URL = "https://dados.agricultura.gov.br/dataset/baefdc68-9bad-4204-83e8-f2888b79ab48/resource/3360a882-f13e-4dc7-aaaa-a1941e950dbc/download/psrdadosabertos2006a2015csv.csv"

def download_file(url, local_path):
    # Open the url image, set stream to True, this will return the stream content.
    with urllib.request.urlopen(url) as url:
        s = url.read()

    # Open a local file with wb ( write binary ) permission.
    with open(local_path, 'wb') as f:
        f.write(s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and persist the data from URL onto a local path.")
    parser.add_argument('local_path', help='The local path to persist the data')
    args = parser.parse_args()
    local_path = args.local_path

    if not os.path.exists(local_path):
        os.makedirs(local_path)

    download_file(URL, local_path + "/psrdadosabertos2006a2015csv.csv")
    print(f"File saved at {local_path}")
