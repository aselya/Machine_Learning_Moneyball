#/PycharmProjects/lahmenCSVmanipulation/csv_files/stats_like_iris.csv

from sklearn.model_selection import _split

import pandas as pd

data = pd.read_csv("/PycharmProjects/lahmenCSVmanipulation/csv_files/team_stats_ave_no_excess_stats.csv")
data_noNA = data.fillna(1)

pop1 , pop2 = _split.train_test_split(data_noNA)

print(pop1)
print(type(pop1))
print(pop2)
print(type(pop2))

pop1.to_csv('/PycharmProjects/lahmenCSVmanipulation/csv_files/csv_data.csv', index=False)
pop2.to_csv('/PycharmProjects/lahmenCSVmanipulation/csv_files/csv_test.csv', index=False)






