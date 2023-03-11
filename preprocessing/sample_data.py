import pandas as pd


PRD_DATA = "data/processed/full_psr_dados_abertos.csv"
PRD_SAMPLE_DATA = "data/processed/psr_soja_pr_sample.csv"


if __name__ == "__main__":
    data = pd.read_csv(PRD_DATA)
    data.query("sg_uf_propriedade=='PR' and nm_cultura_global=='Soja'").sample(
        frac=1
    ).to_csv(PRD_SAMPLE_DATA, index=False)
