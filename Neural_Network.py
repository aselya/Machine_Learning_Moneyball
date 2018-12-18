
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#gets the file for the methods stored in Neural_Net_Data_Methods
import Neural_Net_Data_Methods

#parameters
BATCH_SIZE = 300
TRAIN_STEPS = 3000

#prints log to console
#good for making sure it's running
tf.logging.set_verbosity(tf.logging.INFO)

#the actual neural net is contained here
def make_neural_net(my_feature_columns):
    neural_net =tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1,l1_regularization_strength=0.1),
        # 3 hidden layers of 256 nodes each.
        hidden_units=[256, 256, 256],
        #activation function
        activation_fn=tf.nn.relu,
        #adds dropout
        dropout=0.3,
        #classes coorespond to  number of team values: bad, average, good and playoff
        n_classes=4)
    return neural_net

#returns the train and test vars set up in Neural_Net_Data_Methods file
def get_the_data():
    (train_x, train_y), (test_x, test_y) = Neural_Net_Data_Methods.load_data()
    return (train_x, train_y), (test_x, test_y)

#chooses what columns will be used
def set_feature_columns( train_x):
        my_feature_columns = []
        for key in train_x.keys():
            my_feature_columns.append(tf.feature_column.numeric_column(key=key))
        print("my_feature_columns" + str(my_feature_columns))
        return my_feature_columns

#trains the model on the training data and labels
def train_the_model( classifier, train_x, train_y, ):
    classifier.train(
        input_fn=lambda:Neural_Net_Data_Methods.train_input_fn(train_x, train_y,BATCH_SIZE),steps=TRAIN_STEPS)

#compares testing data to actual data for evaluation
def evaluate_the_results(classifier, test_x, test_y):
    eval_result = classifier.evaluate(input_fn=lambda:Neural_Net_Data_Methods.eval_input_fn(test_x, test_y,BATCH_SIZE))
    return eval_result

#takes new data and generates a prediction
def get_prediction(classifier, input_dictionary):

    predictions = classifier.predict(input_fn=lambda:Neural_Net_Data_Methods.eval_input_fn(input_dictionary,labels=None,batch_size=BATCH_SIZE))
    template = ('\nPrediction is "{}" ({:.1f}%), ')

    #iterates through number of rows of data
    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        actual_prediction = Neural_Net_Data_Methods.TEAM_CLASSIFICATIONS[class_id]
        print(template.format(Neural_Net_Data_Methods.TEAM_CLASSIFICATIONS[class_id],
                              100 * probability))
    return actual_prediction


#gests the training and evaluation data
(train_x, train_y), (test_x, test_y)= get_the_data()

#prints the data retrived
#important thing to look for is # of columns match and ratio or rows in train_x to test_x
print("Neural_Net_Data_Methods load" + str(train_x) + str(train_y))
print("Neural_Net_Data_Methods" + str(test_x) + str(test_y))

 # feature columns are set
my_feature_columns = set_feature_columns( train_x)
#builds the neural net using the feature columns
classifier = make_neural_net(my_feature_columns)

#print statments for debugging
#print("train x and y shape" +str(train_x.shape) )
#print(str(train_y.shape))
#print(str(args.batch_size))

#trains the model on the traning data
train_the_model( classifier, train_x, train_y, )

#tests the model performance on test data
eval_result = evaluate_the_results(classifier, test_x, test_y)

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


#dictionaries to make sure get_prediction method is working
predict_x = {
       'R': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'AB': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'H': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        '2B': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        '3B': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'HR': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'BB': [1 , 1.2, 1.222 , 1.11111111 , 1.11111111],
        'SO': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'SB': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'CS': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'HBP': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'SF': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'RA': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'ER': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'ERA': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'CG': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'SHO': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'SV': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'IPouts': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'HA': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'HRA': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'BBA': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'SOA': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'E' : [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'DP': [1, 1.2, 1.222 , 1.11111111 , 1.11111111],
        'FP': [1, 1.2, 1.222 , 1.11111111 , 1.11111111]
    }

#predict_x2 = {'BB': 1.009641, 'SOA': 1.004817, 'RA': 0.969584, 'DP': 0.991801, 'AB': 0.967161, 'SO': 1.030452, 'SB': 1.032536, 'SV': 0.966212, '3B': 1.005646, 'HRA': 0.960336, 'ERA': 1.04304, 'SHO': 1.030958, '2B': 0.9806010000000001, 'R': 0.964825, 'ER': 0.991213, 'FP': 1.03497, 'HBP': 0.967475, 'IPouts': 0.971961, 'CG': 0.997358, 'E': 1.035217, 'HA': 0.973221, 'H': 1.009223, 'CS': 0.981622, 'SF': 0.968128, 'HR': 0.992999, 'BBA': 1.044765}
predict_x3={'ERA': [1.021266], 'SO': [0.989227], 'BBA': [0.970416], 'RA': [0.967489], 'BB': [1.041685], 'SF': [1.024911], 'E': [0.956521], 'HR': [1.043219], 'ER': [0.9599530000000001], 'HA': [0.995577], 'R': [0.970591], 'SHO': [1.008279], 'HRA': [0.981935], 'CG': [1.000704], 'CS': [1.021651], 'AB': [0.969144], 'SV': [1.01259], 'SOA': [1.044072], 'IPouts': [1.041474], 'H': [1.015401], '3B': [0.966425], 'FP': [0.995196], 'HBP': [0.989337], 'SB': [1.015562], '2B': [1.021475], 'DP': [1.028915]}

#confirms the type
#print(type(predict_x3))

#passes dictionary values and returns a prediction
print("Prediction method check:")
predictions = get_prediction(classifier, predict_x3)

