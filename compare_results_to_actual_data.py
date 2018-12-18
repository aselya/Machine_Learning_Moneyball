'''
Takes the original files used to train the Neural network and calculates a fitness score for teams based on the historical record
Sorts the teams by value of fitness score and uses the original database file to restore team name and year that were stripped in previous methods
finds the teams within .5 +/- for the fitness score that is input into the set_score_and_get_simmilar_teams(score) method
Evaluates each team to see if they would have made playoffs shows results in a print statment

print statments are left for debugging purposes
'''


import pandas as pd

#gets the intial data needed for historical teams, this is used by the neural net as well
team_dataframe = pd.read_csv("/Users/aarondavidselya@gmail/PycharmProjects/Machine_Learning_Baseball_Algorithim/team_stats_ave_no_excess_stats.csv", index_col=False)
team_dataframe = team_dataframe.fillna(1)
#creates a scond copy of the dataframe so different operations can be done without interfering with eachother
dataframe_with_scores = team_dataframe


real_life_fittness_scores = []


#sets the fitness score for the historical teams using the same criteria used in the genetic algorithim to generate teams

def set_the_fittness_scores_of_actual_teams():
    row_count = 0
    number_of_rows = len(team_dataframe.index)
    #print("number of rows" + str(number_of_rows))
    while row_count < number_of_rows:
        score = 0
        current_row = team_dataframe.loc[row_count]
        count = 0
        for item in current_row:
            #print(str(item))
            if count == 26:
                base =  item*50 + 50
                #print('classification_base:' + str(base))
                score = score + base
                #makes sure stats where less is more valuable are treated accordingly
            elif count != 7 and count != 9 and count != 12 and count != 13 and count != 14 and count != 19 and count != 20 and count != 21 and count !=22 and count != 23:
                if item > 1:
                    score = score - item**2
                else:
                    score = score + item**2
            else:
                if item < 1:
                    score = score - item**2
                else:
                    score = score + item**2
            count +=1
        real_life_fittness_scores.append(score)
        #print(type(current_row))
        #print(str(current_row))
        row_count += 1

    #print("row count" +str(row_count))
    #print(real_life_fittness_scores)
    #print("fittnes scores length: " +str(len(real_life_fittness_scores)))

set_the_fittness_scores_of_actual_teams()

team_id = []
def add_the_fittness_scores_and_team_ids_to_dataframe():
    dataframe_with_scores['fittness score']= real_life_fittness_scores

    id = 0
    while id < len(team_dataframe.index):
        team_id.append(id)
        id += 1

    dataframe_with_scores['team id']= team_id

add_the_fittness_scores_and_team_ids_to_dataframe()



#print(dataframe_with_scores)

#sorts the dataframe by fittness score in descenfing order
dataframe_with_scores_sorted =dataframe_with_scores.sort_values(by='fittness score', ascending=False)

#print(dataframe_with_scores_sorted)

#this is the database fils that contains the information that was stripped out by the methods needed to train Neural Network
base_team_dataframe = pd.read_csv('/Users/aarondavidselya@gmail/PycharmProjects/Machine_Learning_Baseball_Algorithim/Team_for_comparison.csv')
#yearID,lgID,teamID,franchID,divID,Rank,G,Ghome,W,L,DivWin,WCWin,LgWin,WSWin,R,AB,H,2B,3B,HR,BB,SO,SB,CS,HBP,SF,RA,ER,ERA,CG,SHO,SV,IPouts,HA,HRA,BBA,SOA,E,DP,FP,name,park,attendance,BPF,PPF,teamIDBR,teamIDlahman45,teamIDretro
#index 14-39 used in the no excess stats
#8 wins  9 losses 6 games
print(base_team_dataframe)







nearby_scores = []
#this is the methods that calls the methods for printing results base on score
def set_score_and_get_simmilar_teams(score):
    dfToList = find_nearby_team(score)
    dataFrame_with_teams = get_list_of_team_stats_for_nearby_return_as_datframe(dfToList)
    print_the_results(dataFrame_with_teams)

def find_nearby_team(score):
    criteria = dataframe_with_scores_sorted[ dataframe_with_scores_sorted.iloc[:,27]>= score-.5 ]
    criteria2 = criteria[criteria.iloc[:,27]<= score+.5]
    #print(criteria)
    #print(criteria2)
#slightly_above = dataframe_with_scores_sorted['fittness score'] < score +2 and dataframe_with_scores_sorted['fittness score'] > score -2
#print(slightly_above)
    dfToList = criteria2['team id'].tolist()
    #print(dfToList)
    return dfToList

new_dataframe = pd.DataFrame

def get_list_of_team_stats_for_nearby_return_as_datframe(dfToList):
    list_of_rows = []
    for item in dfToList:
        item = int(item)
        #print("items:")
        #print(item)
        row = base_team_dataframe.iloc[[item]]

        #row = criteria2.iloc[item, :]
        #print(row)
        #print(type(row))
        #print("year:"+str(row.iloc[0, 0]))
        list_of_rows.append(row)

    #print("list of rows")
    #print(list_of_rows)

    dataFrame_with_teams  = pd.concat(list_of_rows)
    #print("dataframe with teams")
    #print(dataFrame_with_teams)

    dataFrame_with_teams = dataFrame_with_teams.reset_index(drop=True)
    return dataFrame_with_teams

list_of_teams_simmilarity=[]
#print("The historical teams with a fittness score closest to the score inputed," +str(score)+", are:" )

def print_the_results(dataFrame_with_teams):
    total = 0
    total_made_playoffs = 0
    for row in dataFrame_with_teams.index:
        total += 1
        #print(type(row))
        #wins divided by games played
        winning_percentage = dataFrame_with_teams.iloc[row, 8]/dataFrame_with_teams.iloc[row, 6]
        #Winning more than 92 wins gives you a chance of making the playoffs that exceeds 90%
        #92wins/162 games in a modern season results in 0.567 winning percentage
        made_playoffs ="would not have likely made the playoffs"
        if winning_percentage > 0.567:
            made_playoffs = "would be likely made the playoffs"
            total_made_playoffs += 1
        print("The "+str(dataFrame_with_teams.iloc[row, 0])+ " "+ str(dataFrame_with_teams.iloc[row, 40]) +" had a winning percentage of " + "{0:.3f}".format(round(winning_percentage,3))  + " "+ str(made_playoffs))
    print("\n\nOf the " + str(total) + " teams with fittness scores near the one generated by the algorithim\n"+ str(total_made_playoffs) +": would have been very likely to make the playoffs in the modern era"    )
    print("For a "+"{0:.3f}".format(round(total_made_playoffs/total,3)*100) +" percent chance of a team with the top score making the playoffs" )
