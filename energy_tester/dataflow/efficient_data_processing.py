import pandas as pd
import numpy as np

from loguru import logger


def create_data(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        np.random.random(n_rows),
        columns=['value']
    )


def get_average(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.groupby(pd.Series(range(100, dataframe.shape[0], 100)))['value'].mean()


def write_data(dataframe, path):
    dataframe.to_csv(path)


if __name__ == "__main__":
    df = create_data(1000)
    df_avg = get_average(df)
    logger.info(f"Created DataFrame of the following data: \n{df_avg}")
    write_data(df_avg, 'test_path.csv')
