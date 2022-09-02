import warnings

def clean_unnecessary_spaces(out_string):
    """
    Some of the data have spaces before punctuation marks that we need to remove.
    This is used for BART/Roberta
    modified from:
    https://towardsdatascience.com/bart-for-paraphrasing-with-simple-transformers-7c9ea3dfdd8c
    """
    if not isinstance(out_string, str):
        warnings.warn(f">>> {out_string} <<< is not a string.")
        out_string = str(out_string)
    out_string = (
        out_string.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" — ", "—")
        .replace(" - ", "-")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
        .replace(" ’ ", "’")
        .replace("“ ", "“")
        .replace(" ”", "”")
        .replace(" ( ", " (")
        .replace(" ) ", ") ")
    )
    return out_string