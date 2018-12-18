'''
Attempt to use KMeans clustering as a way of classifying data
Did not lead to significant findings


'''

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np



def calculate_classification( row ):
    value = row
    value = float(value)
    if value < float(1):
        return 'Below_Ave_playoff_team'
    else:
        return 'Above_Ave_playoff_team'

def determine_classification(dataFrame):
    df = dataFrame
    df = df.apply (lambda row: calculate_classification( row ) )
    df.head(15)
    df.columns =['Classification']


    return df

def build_and_plot_model():
    dataframe = pd.read_csv('/PycharmProjects/Machine_Learning_Baseball_Algorithim/csv_files/output.csv', index_col=False)
    count_row = str(dataframe.shape[0])  # gives number of row count

    x = dataframe.drop('score', axis=1)
    y = dataframe['score']

    y_classification = determine_classification(y)
    print("test of classification")
    for row in y_classification:
        print(row)

    print("y_classification\n" + str(y_classification))

    print(x)
    print(y)

    x.columns= ['R','AB','H','2B','3B','HR','BB','SO','SB','CS','HBP','SF','RA','ER','ERA','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E' ,'DP','FP']
    y.coumns = ['score']

    model = KMeans(n_clusters=2)
    model.fit(x)

    model.labels_
    print(str(model.labels_))

    plt.figure(figsize=(14,7))

    colormap = np.array(['red', 'lime'])

#plt.subplot(1,2,2)
#plt.scatter(x.ERA, x.SHO, c=colormap[y_classification.values], s=40)
#plt.show()

#plt.subplot(1,2,2)
    #makes a scatter plot comparing the HR and R statistics
    plt.scatter(x.R, x.HR, c=colormap[model.labels_], s=40)
    plt.ylabel("Home Runs")
    plt.xlabel("Runs")
    plt.title("KMeans clustering using HR and Runs")
    save_string = 'KMeans_attempt_sample_size='+ count_row + '.png'
    plt.savefig(save_string)
#plt.scatter(x.H, x.HR, c=colormap[model.labels_], s=40)
    plt.show()


#build_and_plot_model()
