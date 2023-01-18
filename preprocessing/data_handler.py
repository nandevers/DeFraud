
import re

def convert_cols_to_float(data, cols):
    for col in cols:
        data[col] = data[col].apply(lambda x: float(x.replace(",", "")))
    return data



def filter_columns(data, patterns):
    """Return a list of column names in the data frame that match the given pattern.

    Parameters
    ----------
    data : pandas.DataFrame
        The data frame to filter.
    pattern : list of str
        The string patterns to match.

    Returns
    -------
    list
        A list of column names.
    """
    return [col for col in data.columns if any(re.search(p, col) for p in patterns)]

