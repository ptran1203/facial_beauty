import const
import pandas as pd
# import random

def dataframe():
    df = pd.read_csv(const.LABELS_FILE)
    file_names = [const.IMAGES_DIR + file_name for \
                 file_name in df['filename']]
    df['score'] = [str(score) for \
                  score in df['score']]
    return df


def split(df ,rate):
    size = int(len(df) * rate)
    df1 = df.iloc[:size, :]
    df2 = df.iloc[size:, :]
    return (df1, df2)
