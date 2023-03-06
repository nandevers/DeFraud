import json
import os
import re
from pathlib import Path

import pandas as pd
from constants import RAW_COL_TYPES
from data_handler import filter_columns
from unidecode import unidecode

RAW_DATA = Path("data/raw/")
PRD_DATA = Path("data/processed/")


def group_names(data):
    # come up with groups of columns that need data transformation
    lat_long_cols = filter_columns(data, ["lat", "lon"])
    date_cols = filter_columns(data, ["^dt", "data"])
    int_cols = lat_long_cols + ["nr_animal", "ano_apolice"]
    str_cols = filter_columns(data, ["nm", "event", "sg_uf_propriedade"])
    code_id_cols = filter_columns(data, ["^nr", "^id", "^cd"])
    code_id_cols = [
        col
        for col in code_id_cols
        if col
        not in lat_long_cols
        + ["nr_area_total", "nr_animal", "nr_produtividade_segurada"]
    ]
    str_cols = str_cols + code_id_cols

    col_map = str_cols + code_id_cols + date_cols + int_cols + lat_long_cols
    value_cols = list(set(data.columns.to_list()) - set(col_map))
    # TODO: this is tooooo ugly
    cols_dict = {
        "str_cols": str_cols,
        "code_id_cols": code_id_cols,
        "int_cols": int_cols,
        "date_cols": date_cols,
        "lat_long_cols": lat_long_cols,
        "value_columns": value_cols,
    }
    file_path = "cols_dict.json"
    with open(file_path, "w") as f:
        json.dump(cols_dict, f, indent=4)
    return str_cols, value_cols, date_cols


def read_chunk(path):
    return pd.concat(
        [
            chunk.replace({"-": "", "%": "", ",": "."}, regex=True)
            for chunk in pd.read_csv(
                path,
                delimiter=";",
                decimal=",",
                encoding="latin-1",
                dtype=RAW_COL_TYPES,
                na_values=["-"],
                skipinitialspace=True,
                chunksize=10000,
            )
        ]
    )


def output_cols(data_list: [pd.DataFrame]):
    all_cols = []
    [all_cols.extend(d.columns.to_list()) for d in data_list]
    unique_cols = set(all_cols)
    return list(unique_cols)


def clean_data(data):
    data["niveldecobertura"] = (data["niveldecobertura"].fillna("9999")).astype(float)
    data.loc[data["niveldecobertura"].astype(float) > 10, "niveldecobertura"] = (
        data.loc[data["niveldecobertura"].astype(float) > 10, "niveldecobertura"] / 100
    )
    data["formal_latitude"] = (
        data["nr_grau_lat"].astype(str)
        + "° "
        + data["nr_min_lat"].astype(str)
        + "' "
        + data["nr_seg_lat"].astype(str)
        + "'' S"
    )
    data["formal_longitude"] = (
        data["nr_grau_long"].astype(str)
        + "° "
        + data["nr_min_long"].astype(str)
        + "' "
        + data["nr_seg_long"].astype(str)
        + "'' W"
    )
    return data


def concat_data(data):
    keep_cols = output_cols(data)
    io = []
    for d in data:
        try:
            io.append(d[keep_cols])
        except:
            # Create sets of a,b
            setA = set(keep_cols)
            setB = set(
                [
                    "EVENTO_PREPONDERANTE",
                    "VALOR_INDENIZAÇÃO",
                    "NR_DECIMAL_LONGITUDE",
                    "NR_DECIMAL_LATITUDE",
                ]
            )
            # Get new set with elements that are only in a but not in b
            onlyInA = setA.difference(setB)
            onlyInA = list(onlyInA)
            io.append(d[onlyInA])
    return pd.concat(io, axis=0, ignore_index=True, join="outer")


def parse_date(date):
    try:
        return pd.to_datetime(date, format="%d/%m/%Y")
    except ValueError:
        return pd.to_datetime(date, format="%d/%m/%Y %H:%M")


if __name__ == "__main__":
    raw_files = os.listdir(RAW_DATA)
    df = [read_chunk(RAW_DATA / f) for f in raw_files]

    # Concat datasets
    data = concat_data(df).copy()
    del df

    data.columns = [unidecode(c) for c in data.columns.str.lower()]
    str_cols, value_cols, date_cols = group_names(data)

    # Apply Transformations
    data[str_cols + value_cols] = data[str_cols + value_cols].astype(str)

    ## Clean up data
    data = clean_data(data)
    data["dt_proposta"] = pd.to_datetime(data["dt_proposta"], dayfirst=True)

    ## Write data to predefined path
    if not os.path.exists(PRD_DATA):
        os.makedirs(PRD_DATA)

    data.to_csv(PRD_DATA / "full_psr_dados_abertos.csv", index=False)
