import math


def get_score(tokens, roberta):
    tokens_encoded = roberta.encode(tokens)
    return math.exp(roberta.predict('rescorer_head', tokens_encoded)[:, 1].item())

def output_csv(df, cname, roberta, output_path):
    df['score'] = df[cname].apply(lambda x: get_score(x, roberta))
    df = df.sort_values(by=['score'], ascending=False)
    df.to_csv(output_path, index=False, sep='\t', encoding='utf-8')

