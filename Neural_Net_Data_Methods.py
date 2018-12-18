import tensorflow as tf
from sklearn.model_selection import _split

import pandas as pd

'''
Takes in the sanitized data
fills in any unfilled values with a value of 1
splits the population into testing and regular data csv files
'''
data = pd.read_csv("/PycharmProjects/Machine_Learning_Baseball_Algorithim/team_stats_ave_no_excess_stats.csv")
data_noNA = data.fillna(1)

#splits the csv into seperate populations
pop1 , pop2 = _split.train_test_split(data_noNA)

print(pop1)
print(type(pop1))
print(pop2)
print(type(pop2))


#write populations to CSVs
pop1.to_csv('/PycharmProjects/Machine_Learning_Baseball_Algorithim/csv_files/csv_data.csv', index=False)
pop2.to_csv('/PycharmProjects/Machine_Learning_Baseball_Algorithim/csv_files/csv_test.csv', index=False)




CSV_COLUMN_NAMES = ['R','AB','H','2B','3B','HR','BB','SO','SB','CS','HBP','SF','RA','ER','ERA','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E','DP','FP','Classification']

#     numerical value:        0            1             2               3
TEAM_CLASSIFICATIONS = ['Bad Team', 'Average Team', 'Good Team', 'Playoff Team']


def load_data(y_name='Classification'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""

    train = pd.read_csv('/PycharmProjects/Machine_Learning_Baseball_Algorithim/csv_files/csv_data.csv', names=CSV_COLUMN_NAMES, header=1)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv('/PycharmProjects/Machine_Learning_Baseball_Algorithim/csv_files/csv_test.csv', names=CSV_COLUMN_NAMES, header=1)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


# The remainder of this file contains a simple example of a csv parser,
#     implemented using the `Dataset` class.

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
# last one is an int to reflect classification
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0],[0.0], [0.0], [0.0], [0.0],[0.0], [0.0], [0.0], [0.0],[0.0], [0.0], [0.0], [0.0],[0.0], [0.0], [0.0], [0.0],[0.0], [0.0], [0.0], [0.0],[0.0], [0.0], [0]]

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('Classification')

    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset
