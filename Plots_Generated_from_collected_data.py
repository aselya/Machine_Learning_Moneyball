'''
This file takes the output.csv file generated in the Genetic_Algorithim_Component
and builds a series of plots to help visualize the data generated by the results

'''


import matplotlib.pyplot as plt
import pandas as pd
import kmeans_on_results

#df = pd.read_csv('/PycharmProjects/Machine_Learning_Baseball_Algorithim/csv_files/output.csv')
#count_row = str(df.shape[0])  # gives number of row count



'''
reads the CSV and plots each datapoint as a symbol except score which is plotted as a red line
cool plot but too congested to use for any significant number of samples
'''

def plot_all_the_stats_with_score_as_line():
    df = pd.read_csv('/PycharmProjects/Machine_Learning_Baseball_Algorithim/csv_files/output.csv')
    count_row = str(df.shape[0])
    print("row count:" + str(count_row))
    df.plot(style=['D','D','D','D','D','D','D','D','D','D','s','s','s','s','s','s','s','o','o','o','o','o','o','o','o','o','r'])
    plt.title("Scatter plot with all the stats score as line")
    save_string = '/plots/plot_all_the_stats_with_score_as_line_sample_size='+ str(count_row)+'.png'
    plt.savefig(save_string)
    plt.show()



'''
Gets saved csv and finds the 4 largest values for each top team and saves them to a dataframe
replaces the values in dataframe with the stats associated with each max value
finds counts for each statistic and plots them on a multiline graph

'''
def max_in_row():
    dataframe = pd.read_csv('/PycharmProjects/Machine_Learning_Baseball_Algorithim/csv_files/output.csv', index_col=False)
    count_row = str(dataframe.shape[0])
    df = dataframe.T #transposes
    #performs ranking
    df = df.rank(method='max', ascending=False)
    new_df = pd.DataFrame()
    #gets top 4 for each row
    for i in df.columns:
        s = df[i][df[i]<5].sort_values().reset_index().drop(i, axis=1)
        new_df = pd.concat([new_df, s.T])

    #reassigns index sets column values and index column name for reference
    new_df.index = df.columns
    new_df.index.name = "stats"
    new_df.columns = ["Most", "2nd Most", "3rd Most", "4th Most"]

    #prints for debugging
    print("Most Common")
    print(new_df)
    print("value counts\n"+str(new_df.apply(pd.value_counts)))
    print(type(new_df))

    #sets the count values
    count_df = new_df.apply(pd.value_counts)
    #plots the values
    count_df.plot.bar()
    #sets the plot parameters
    plt.title('The most common statistics')
    plt.ylabel("Number Of Times")
    plt.xlabel("Statistics")

    save_string = '/plots/max_in_row_sample_size='+str(count_row) + '.png'
    plt.savefig(save_string)

    plt.show()


'''
Repeat logic from max_in_row() but for smallest values
Gets saved csv and finds the 4 largest values for each top team and saves them to a dataframe
replaces the values in dataframe with the stats associated with each max value
finds counts for each statistic and plots them on a multiline graph

'''
def min_in_row():
    dataframe = pd.read_csv('/PycharmProjects/Machine_Learning_Baseball_Algorithim/csv_files/output.csv', index_col=False)
    count_row = str(dataframe.shape[0])

    df = dataframe.T

    df = df.rank(method='min', ascending=True)
    new_df = pd.DataFrame()

    for i in df.columns:
        s = df[i][df[i]<5].sort_values().reset_index().drop(i, axis=1)
        new_df = pd.concat([new_df, s.T])

    new_df.index = df.columns

    new_df.index = df.columns
    new_df.index.name = "stats"
    new_df.columns = ["Least", "2nd Least", "3rd Least", "4th Least"]


    print(new_df)
    print("low value counts\n"+str(new_df.apply(pd.value_counts)))
    count_df = new_df.apply(pd.value_counts)
    count_df.plot.bar()
    plt.title('The Least common statistics')
    plt.ylabel("Number Of Times")
    plt.xlabel("Statistics")

    save_string = '/plots/min_in_row_size='+str(count_row) + '.png'
    plt.savefig(save_string)
    plt.show()


'''
prints a bar graph of the average value
has line at 0 to represent average
'''
def bar_graph_average():
    dataframe = pd.read_csv('/PycharmProjects/Machine_Learning_Baseball_Algorithim/csv_files/output.csv', index_col=False)
    count_row = str(dataframe.shape[0])

    df = dataframe.mean(axis=0)
    #subtracting 1 from value makes it so y = 0 is the average not y= 1
    df = df.sub(1, axis=0)
    print(str(type(df)))
    #ax = df.plot(kind='bar')
    ax =df.plot(kind='bar')
    #line on y =0 for visual effect
    ax.axhline(y= 0, color='g', linestyle='--', lw=2)

    #lines added to show significant varation
    ax.axhline(y= 0.05, color='r', linestyle='--', lw=2)
    ax.axhline(y= -0.05, color='r', linestyle='--', lw=2)
    ax.set_title("Average of the stats for each generations best team")

    save_string = '/plots/bar_graph_average_sample_size='+str(count_row) + '.png'
    plt.savefig(save_string)

    plt.show()


'''
prints a bar graph of the std devation values
has line at 0 to represent average
leaving out the score because stats are what we want
'''
def bar_graph_std():
    dataframe = pd.read_csv('/PycharmProjects/Machine_Learning_Baseball_Algorithim/csv_files/output.csv', index_col=False)
    count_row = str(dataframe.shape[0])

    df = dataframe.std(axis=0)
    #gets all but score since we want stats
    df= df[:-1]
    print(str(type(df)))
    ax =df.plot(kind='bar')
    ax.axhline(y= 0, color='g', linestyle='--', lw=2)
    ax.axhline(y= 0.3, color='r', linestyle='--', lw=2)
    #ax.axhline(y= -0.05, color='r', linestyle='--', lw=2)
    ax.set_title("Standard Deviation of the stats for each generations best team")

    save_string = '/plots/bar_graph_std_sample_size='+str(count_row) + '.png'
    plt.savefig(save_string)

    plt.show()

    print("index length: "+str(len(df.index)))


#methods for calling the various plots

def build_Plots():
   # plot_all_the_stats_with_score_as_line()

    #kmeans_on_results.build_and_plot_model()

    max_in_row()

    min_in_row()

    bar_graph_average()

    bar_graph_std()

build_Plots()
