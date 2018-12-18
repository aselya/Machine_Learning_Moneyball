import pandas as pd
import random
import Neural_Network #imports the neural net file
from Neural_Network import classifier
#import Plots_Generated_from_collected_data #imports file
import compare_results_to_actual_data

#classifier = '/Users/aarondavidselya@gmail/PycharmProjects/Machine_Learning_Baseball_Algorithim/Neural_Network_saved'

'''

since the goal is to find the stats that make the most difference while still making the playoffs
points will be rewarded for any stat below average since that indicates the stat has no significant impact upon wins
and shouldn't be what teams focus on
conversely any stat below average will be rewarded with a higher contribution fitness score
The base rates are far apart enough that there shouldn't be a chance for a bad team to receive a high enough score to be a playoff team
ideal is 1
for each stat if the stat is > 1
    take difference squared between stat and ideal and subtract from base score
for each stat <= 1
    take difference squared from stat and ideal and add it to the base score

Some stats benefit from beineg lower and their evaluation will be inverse of the above function

starting breeding population will be 10 randomly chosen from existing data
top 2 will be parents

fitness equation base scores to encourage playoff teams based on winning percentage
likely playoff team = 200
likely good team = 150
likely average team = 100
likely bad team = 50

print statments have been left in from debugging by are commented out

'''
GENERATIONS = 10 #sets the main loop
#dataset compiled from previous operations in Data_extraction_and_sanitization.py
DATASET = pd.read_csv('/Users/aarondavidselya@gmail/PycharmProjects/Machine_Learning_Baseball_Algorithim/team_stats_ave_no_excess_stats.csv', index_col=None)
POPULATION_TEAMS = []
IDEAL_POPULATION_SIZE = 10
MUTATION_RATE = 0.05
MAKE_AVE_RATE = 0.2

def get_number_of_stats_tracked():
    random_subset = DATASET.sample(n=IDEAL_POPULATION_SIZE)
    #drop lable column, that will be determined by NN
    random_subset1 = random_subset.drop(columns=['Classification'])
    #print("columns" + str(len(random_subset1.columns)))
    NUMBER_OF_STATS_TRACKED = len(random_subset1.columns)
    #print("number stat tracked: " + str(NUMBER_OF_STATS_TRACKED))
    #print(random_subset1.head(10))
    return random_subset1

BASE_POPULATION = get_number_of_stats_tracked()
#print("base population length: " + str(len(BASE_POPULATION)))
NUMBER_OF_STATS_TRACKED = len(BASE_POPULATION.columns)
#print("number of stats tracked" + str(NUMBER_OF_STATS_TRACKED))


'''
class contains the stat values for a single team in the population
Each stat is treated as if they are genes in the chromosome
'''
class Team_chromosomes:
    #generates a new statistical value
    def make_new_stat_value(self):
        #1 represents the average value of a team
        #the selection of a randomint is added to 1 to create a stat that can vary around the average
        new_value = 1 + random.randint(-60000000,60000000)/100000000
        #print("new stat value" + str(new_value))
        return new_value
    def make_ave_stat_value(self):
        new_value = 1.0
        return new_value

    #returns the stat genes in the form of a dictionary
    #so the neural net can make a prediction
    def format_chromosome_as_dict(self, chromosome):
        ("format_chromosome_as_dict")
        default_value = 1
        #keys to dictionary are the samea stats that are tracked by the neural net
        keys = ['R','AB','H','2B','3B','HR','BB','SO','SB','CS','HBP','SF','RA','ER','ERA','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E' ,'DP','FP']
        dictionary = dict.fromkeys(['R','AB','H','2B','3B','HR','BB','SO','SB','CS','HBP','SF','RA','ER','ERA','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E' ,'DP','FP'],default_value)
        count = 0
        for gene in chromosome:
            dictionary[keys[count]] = [gene]
            count += 1
        #print("dictionary" + str(dictionary))

        return dictionary
    #passes the dictionary generated in format_chromosome_as_dict
    #to the neural network and returns classification prediction
    def get_nn_classification(self, dict):
        pred = Neural_Network.get_prediction(classifier, dict)
        #print(pred)
        print("prediction:" + str(pred))
        return pred



    '''
    in most cases having a stat that is above average will benefit the team
    However, in baseball some stats are more beneficial to be lower than higher
    An adjustment to the way those stats contribute to fittness score is needed
    are all such stats and the method below corrects for them
    '''
    def lower_than_average_benefits_fittness_score( gene, score):
        if gene < 1:
            score = score - gene**2
        else:
            score = score + gene**2
        return score

    #uses the Neural Net prediction and stats in chromosoeme to set fitness score
    #Base value is set by team prediction: Playoff team > Good Team > Average Team > Bad Team
    #Any stat below average (1) provides a positive benefit to fittness score
    #Any stat above average (1) provides a negitive contribution to fittness score
    #The balancing is to prevent the algorithim from just generating an ever increasing value for each stat
    def set_fitness_score(self, chromosome, NN_prediction):
        #print("set_fittness_score" + str(NN_prediction))
        prediction_string = str(NN_prediction)
        score = 0
        count = 0
        if prediction_string == "Bad Team":
            score = score + 50
        if prediction_string == "Average Team":
            score = score +100
        if prediction_string == "Good Team":
            score = score +150
        if prediction_string == "Playoff Team":
            score = score +200

        #print("set score:")

        for gene in chromosome:
            #results were skewing towards significantly lower number of at bats which is not consistant with real baseball
                #this reduces the possible variance to a more reasonable level
            if count == 1:
                score = score + 1 + random.randint(-100,100)/1000

            #  SO -stike outs by batters, CS-caught stealing, RA-runs allowed, ER-Earned Runs, ERA, HA-hits allowed, HRA-hr allowed, BBA- BB allowed and E-errors
            elif count != 7 and count != 9 and count != 12 and count != 13 and count != 14 and count != 19 and count != 20 and count != 21 and count !=22 and count != 23  :
                if gene > 1:
                    score = score - gene**2
                else:
                    score = score + gene**2
            else:
                score = self.lower_than_average_benefits_fittness_score(gene, score)
        #print("score is : " + str(score))
        count += 1
        return score

    #constructor for the chromosome class
    #sets intiail values and calls methods
    def __init__(self, length):
        self.genes_made_of_stats = []
        self.fittness_score = 0
        self.current_length = 0
        self.fittness_score = 0

        #print("new chromosome created")
        #print("length" + str(length))
        #print("current length :" + str(self.current_length) + " numb stats tracked" + str(length))
        while self.current_length < length:
            new_gene = self.make_new_stat_value()
            self.genes_made_of_stats.append(new_gene)
            self.current_length += 1
        prediction = self.get_nn_classification(self.format_chromosome_as_dict(self.genes_made_of_stats))

        self.fittness_score = self.set_fitness_score(self.genes_made_of_stats, prediction)
#end chromosome class

#copy of a previous method for accessing outside of the chromosome class
def format_chromosome_as_dict(chromosome):
        #print("format_chromosome_as_dict")
        default_value = 1
        keys = ['R','AB','H','2B','3B','HR','BB','SO','SB','CS','HBP','SF','RA','ER','ERA','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E' ,'DP','FP']
        dictionary = dict.fromkeys(['R','AB','H','2B','3B','HR','BB','SO','SB','CS','HBP','SF','RA','ER','ERA','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E' ,'DP','FP'],default_value)
        count = 0
        for gene in chromosome.genes_made_of_stats:
            dictionary[keys[count]] = [gene]
            count += 1
        #print("dictionary" + str(dictionary))

        return dictionary

#one point crossover in the middle from 2 parents
#impliments mutation as well as binary selection of genes
def cross_over( parent1, parent2):
    #makes new chromosome that will be filled with genes from parents
    new_team_chromosome = Team_chromosomes(NUMBER_OF_STATS_TRACKED)
    count = 0
    for gene in new_team_chromosome.genes_made_of_stats:
        if count > NUMBER_OF_STATS_TRACKED/2:
            new_team_chromosome.genes_made_of_stats[count] = parent1.genes_made_of_stats[count]
        else:
            new_team_chromosome.genes_made_of_stats[count] = parent2.genes_made_of_stats[count]
            #mutation is occuring here
        mutation_indicator = random.random()
        if mutation_indicator < MUTATION_RATE:
           new_team_chromosome.genes_made_of_stats[count] = new_team_chromosome.make_new_stat_value()
        #this section takes values and replaces them with a 1.0
        #since in this model 1 is a value that does not contribute positivly or negativly to fitness score
        #it simulates a GA fed by a binary selection of genes
        make_ave = random.random()
        if make_ave < MAKE_AVE_RATE:
            new_team_chromosome.genes_made_of_stats[count] = new_team_chromosome.make_ave_stat_value()
        count += 1

    #gets neural net classification for new child chromosome

    prediction = new_team_chromosome.get_nn_classification(new_team_chromosome.format_chromosome_as_dict(new_team_chromosome.genes_made_of_stats))
    #sets fittness score for newly created chromosome
    new_team_chromosome.set_fitness_score(new_team_chromosome.genes_made_of_stats, prediction)

    format_chromosome_as_dict(new_team_chromosome)
    return new_team_chromosome

    #builds the population that the Gentic Algorithim will be implimented on
def build_POPULATION_TEAMS( population):
    #stores the new population
    NEW_POPULATION_TEAMS = []
    #loops till population is full
    while len(NEW_POPULATION_TEAMS) < IDEAL_POPULATION_SIZE:
        #print("numbrer of stats tracked " +str(NUMBER_OF_STATS_TRACKED))
        NEW_POPULATION_TEAMS.append(Team_chromosomes(NUMBER_OF_STATS_TRACKED))
    #sorts population so largest fittness score is first
    NEW_POPULATION_TEAMS.sort(key=lambda x: x.fittness_score, reverse=True)
    #NEW_POPULATION_TEAM = NEW_POPULATION_TEAMS[0:4] #moved here from below for loop to see if it makes a difference

    for chrom in POPULATION_TEAMS:
        #print("at begining of for loop"+str(chrom.fittness_score))

        #commented code used for larger population size with more than 2 parents
        '''
        for using more than top 2 parent selection for crossove
        NEW_POPULATION_TEAM = NEW_POPULATION_TEAMS[0:4]
        randint = random.randint(0,4)
        randint2 = random.randint(0,4)
        while randint == randint2:
            randint = random.randint(0,4)
        NEW_POPULATION_TEAMS.append(cross_over(NEW_POPULATION_TEAMS[randint], NEW_POPULATION_TEAMS[randint2]))

        '''
        #using just 2 parents
        NEW_POPULATION_TEAM = NEW_POPULATION_TEAMS[0:2]
        NEW_POPULATION_TEAMS.append(cross_over(NEW_POPULATION_TEAMS[0], NEW_POPULATION_TEAMS[1]))

        #print("at end of for loop"+ str(chrom.fittness_score))
    return NEW_POPULATION_TEAMS

dataFrame_with_results = pd.DataFrame(columns= ['R','AB','H','2B','3B','HR','BB','SO','SB','CS','HBP','SF','RA','ER','ERA','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E' ,'DP','FP'])

def take_dicts_form_dataframe (dict):
    #print("take_dicts_form_dataframe")
    updated_dataFrame = dataFrame_with_results

    updated_dataFrame =updated_dataFrame.append(pd.DataFrame.from_dict(dict))

    #print("test:" + str(updated_dataFrame['ERA'].values))
    return updated_dataFrame

'''
main loop that goes until the number of generations has been reached
'''

count = 0 #tracks generation loops
TOP_SCORES = [] #stores each genrations top scores
TOP_TEAMS = []
BEST_SCORE = 0 #stores the best score
while count < GENERATIONS:
    #builds a new population each loop
    POPULATION_TEAMS = build_POPULATION_TEAMS(POPULATION_TEAMS)
    count += 1
    print("generations:" + str(count))
    #sorts population so largest fittness score is first
    POPULATION_TEAMS.sort(key=lambda x: x.fittness_score, reverse=True)
    #removes any additional population that may have been appended
    POPULATION_TEAMS = POPULATION_TEAMS[0:IDEAL_POPULATION_SIZE]

    #print(str(POPULATION_TEAMS))
    print_count = 0
    for chrom in POPULATION_TEAMS:
        if print_count == 0:
            TOP_SCORES.append(chrom.fittness_score)
            #print("top score updated")
            TOP_TEAMS.append(chrom)

            if BEST_SCORE < chrom.fittness_score:
                BEST_SCORE = chrom.fittness_score
                #print("best score updated:" + str(chrom.fittness_score))
        print(str(chrom.fittness_score))
        print_count += 1


    gen_count_for_print = 1
    for score in TOP_SCORES:
        print('Top Score for Generation '+str(gen_count_for_print) +': ' + str(score) )
        gen_count_for_print += 1
    top_team_gen_count = 1


'''
I tried to put this loop and write command in a method but for some reason it just won't work
do not touch it!
'''
for chrom in TOP_TEAMS:
    dataFrame_with_results=take_dicts_form_dataframe(format_chromosome_as_dict(chrom))

#compares the fittnes scores to the average to normalize them
def get_normalized_fitness_scores(list):
    mean = sum(list)/len(list)
    count = 0
    #print("mean:" + str(mean))
    normalized_fitness_scores=[]
    while count < len(list):
        normalized_fitness_scores.append(list[count]/mean)
        count += 1
    return normalized_fitness_scores

#appends the normalized scores to the dataFrame_with_results and writes them to the output.csv file
dataFrame_with_results['score']= get_normalized_fitness_scores(TOP_SCORES)
with open("/Users/aarondavidselya@gmail/PycharmProjects/Machine_Learning_Baseball_Algorithim/output.csv", 'a') as f:
            dataFrame_with_results.to_csv(f, index=None)


#Calls the build_Plot method from the imported file
#Plots_Generated_from_collected_data.build_Plots()

print("The algorithim generated a team with the top score of" + "{0:.4f}".format(round(TOP_SCORES[0],4))+ ".\nA similar score was, achieved by the following historical teams:\n")
compare_results_to_actual_data.set_score_and_get_simmilar_teams(TOP_SCORES[0])
