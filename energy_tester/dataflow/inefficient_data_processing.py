import pandas as pd
import numpy as np

from loguru import logger


def create_data(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        np.random.random(n_rows),
        columns=['value']
    )


def get_brute_force_average(dataframe: pd.DataFrame) -> pd.DataFrame:
    accumulator = 0
    df_builder = pd.DataFrame()
    for index, row in dataframe.iterrows():
        accumulator += row.value
        if index % 100 == 0 and index != 0:
            df_builder = pd.concat([df_builder, (pd.DataFrame([accumulator / 100], index=[index], columns=['value']))])
            accumulator = 0

    return df_builder


def write_data(dataframe, path):
    dataframe.to_csv(path)


if __name__ == "__main__":
    df = create_data(1000)
    df_avg = get_brute_force_average(df)
    logger.info(f"Created DataFrame of the following data: \n{df_avg}")
    write_data(df_avg, 'test_path.csv')
