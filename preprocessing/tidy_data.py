import json

import pandas as pd
from constants import RAW_COL_TYPES
from data_handler import convert_cols_to_float, filter_columns

RAW_DATA = "data/raw/psrdadosabertos2006a2015csv.csv"
PRD_FOLDER = "data/processed/"
PRD_DATA = "data/processed/psrdadosabertos2006a2015csv.csv"


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


if __name__ == "__main__":
    data = pd.concat(
        [
            chunk.replace("-", "")
            for chunk in pd.read_csv(
                RAW_DATA,
                delimiter=";",
                encoding="latin-1",
                dtype=RAW_COL_TYPES,
                na_values=["-"],
                skipinitialspace=True,
                chunksize=10000,
            )
        ]
    )
    data.columns = data.columns.str.lower()
    str_cols, value_cols, date_cols = group_names(data)

    # Apply Transformations
    data[str_cols] = data[str_cols].astype(str)
    data[value_cols] = data[value_cols].apply(
        lambda x: x.str.replace(",", "").astype(float)
    )
    data[date_cols] = data[date_cols].apply(pd.to_datetime, format="%d/%m/%Y")
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

    # Write data to predefined path
    data.to_csv(PRD_DATA, index=False)
