import json

import pandas as pd
from f_utils import CumulativeCountEncoder, CumulativeEncoder

PRD_SAMPLE_DATA = "../../data/processed/psr_sample.csv"
PRD_FEAT_DATA = "../../data/processed/psr_featues.csv"
PRD_SAMP_FEAT_DATA = "../../data/processed/psr_sampled_featues.csv"

if __name__ == "__main__":
    # load column dict
    with open("cols_dict.json") as f:
        cols_dict = json.load(f)

    # load data
    data = pd.read_csv(PRD_SAMPLE_DATA)
    tdata = []

    # cumulative count encoding
    transformer = CumulativeCountEncoder(
        group_by_cols=["nr_documento_segurado"], sort_col="dt_proposta"
    )
    transformer.fit(data)
    data_transformed = transformer.transform(data)
    print(data_transformed.shape)

    # cumulative operation encoding
    for c in cols_dict["value_columns"]:
        encoder = CumulativeEncoder(
            operation=["max", "min", "sum"],
            col=c,
            group_by_cols=["nr_documento_segurado"],
            sort_col="dt_proposta",
        )
        encoder.fit(data)
        td = encoder.transform(data)
        print(td.shape)
        tdata.append(td)

    data = pd.concat(tdata, axis=1)
    print(data.shape)
    data = pd.concat([data, data_transformed.iloc[:, -1]], axis=1)
    print(data.shape)
    data = data.loc[:, ~data.columns.duplicated()]
    print(data.shape)
    # save to csv
    data.to_csv(PRD_FEAT_DATA, index=False)

    # data = data.query("valor_indenização>100").loc[:, "second_digit"].unique()
    write_cols = [
        "nr_documento_segurado",
        "nr_proposta",
        "dt_proposta",
        "evento_preponderante",
        "valor_indenização",
    ] + data.columns.to_list()[-25:]

    data[write_cols].to_csv(PRD_SAMP_FEAT_DATA, index=False)
