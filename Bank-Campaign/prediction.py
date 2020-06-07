
from data_lib import *

DATASET_PATH = 'bank-additional-full.csv'

# Exploration of DATA

if __name__ == "__main__":
    # Preprocessing #
    df = load_dataset(dataset_path=DATASET_PATH, separator=';')
    # general_info(df)
    displayHead(df, True, True)
    # print(missing_values_percentage(df)) # None
    # print(missing_rate(df)) # 0% missing in each category
    # print(analyse_target(df, 'y', normalized=True))
    # draw_histograms(df, data_type='float64')
    # print(description_object(df, 'y'))
    """
    Target/Variables relation :
    """
    # Yes and No subsets
    yes_df = subset_creator(df, 'y', 'yes')
    no_df = subset_creator(df, 'y', 'no')
    contact_column = df['contact']
    telephone_df = subset_creator(df, 'contact', 'telephone')
    cellular_df = subset_creator(df, 'contact', 'cellular')

    relation = [(yes_df, 'yes'), (no_df, 'no')]
    display_relations(telephone_df, relation)

    # cell_df = qual_to_quan(df, "contact", 'cellular')
    # phone_df = qual_to_quan(df, "contact", 'telephone')
    # relation = [(cell_df, 'cellular'), (phone_df, 'telephone')]
    # y_column = qual_to_quan(df, "y", 'yes')
    # display_relations(cell_df, relation)
