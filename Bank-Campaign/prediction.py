
from data_lib import *

DATASET_PATH = 'bank-additional-full.csv'

# Exploration of DATA

if __name__ == "__main__":
    # Preprocessing #
    df = load_dataset(dataset_path=DATASET_PATH, separator=';')
    # general_info(df)
    # displayHead(df, True, True)
    # print(missing_values_percentage(df)) # None
    # print(missing_rate(df)) # 0% missing in each category
    # print(analyse_target(df, 'y', normalized=True))
    draw_histograms(df, data_type='float64')
