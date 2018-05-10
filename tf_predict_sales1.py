#do required imports
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import os.path

#model parameters
learning_rate = 0.001
training_epochs = 200
display_step = 5

#load trainig data
training_data_tf = pd.read_csv("square_root_training.csv", dtype=float)

training_data_x = training_data_tf.drop('actual_result', axis=1).values
training_data_y = training_data_tf[['actual_result']].values

#load testing data
testing_data_tf = pd.read_csv("square_root_testing.csv", dtype=float)

testing_data_x = testing_data_tf.drop('actual_result', axis=1).values
testing_data_y = testing_data_tf[['actual_result']].values

#scalers from scikit learn
x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))

#now scaling data
training_data_x_scaled = x_scaler.fit_transform(training_data_x)
training_data_y_scaled = y_scaler.fit_transform(training_data_y)

testing_data_x_scaled = x_scaler.transform(testing_data_x)
testing_data_y_scaled = y_scaler.transform(testing_data_y)


#model neural network
#number of input and output nodes
ann_input_nodes = 1
ann_output_nodes = 1

#nodes in different layers
ann_nodes_layer1 = 50
ann_nodes_layer2 = 100
ann_nodes_layer3 = 25

#define the layes

#Input layer
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None,ann_input_nodes))

#Layer 1
with tf.variable_scope('layer_1'):
    weights = tf.get_variable(name="weights1", shape=[ann_input_nodes, ann_nodes_layer1], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases1', shape=[ann_nodes_layer1], initializer=tf.zeros_initializer())
    layer1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

#Layer 2
with tf.variable_scope('layer_2'):
    weights = tf.get_variable(name="weights2", shape=[ann_nodes_layer1, ann_nodes_layer2], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases2', shape=[ann_nodes_layer2], initializer=tf.zeros_initializer())
    layer2_output = tf.nn.relu(tf.matmul(layer1_output, weights) + biases)

#Layer 3
with tf.variable_scope('layer_3'):
    weights = tf.get_variable(name="weights3", shape=[ann_nodes_layer2, ann_nodes_layer3], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases3', shape=[ann_nodes_layer3], initializer=tf.zeros_initializer())
    layer3_output = tf.nn.relu(tf.matmul(layer2_output, weights) + biases)

#Output Layer
with tf.variable_scope('output_layer'):
    weights = tf.get_variable(name="weights_output", shape=[ann_nodes_layer3, ann_output_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases_output', shape=[ann_output_nodes], initializer=tf.zeros_initializer())
    prediction = tf.nn.relu(tf.matmul(layer3_output, weights) + biases)

#Cost function
with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None, 1))
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

#Optimizer or Minimizer
with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    #alternate optimizer
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#logging
with tf.variable_scope('logging'):
    tf.summary.scalar('current_cost', cost)
    summary = tf.summary.merge_all()

#model saver
saver = tf.train.Saver()

#Initialize the training loop
with tf.Session() as session:

    #try to load existing model if it exists.
    #in that case, do not run the training loop
    if os.path.exists("./logs/checkpoint"):
        saver.restore(session, "./logs/sqrt_model.ckpt")
        print("loaded previously saved model")
    else:
        print("training loop initialized")
        #initialize global variables
        session.run(tf.global_variables_initializer())

        #writing logs for visualization
        training_log_writer = tf.summary.FileWriter("./logs/training", session.graph)
        testing_log_writer = tf.summary.FileWriter("./logs/testing", session.graph)

        #run the training data over and over through the network
        #one epoch is equal to 1 pass of all the training data through the network
        for epoch in range(training_epochs):

            #feed in training data and do one pass
            session.run(optimizer, feed_dict={X: training_data_x_scaled, Y: training_data_y_scaled})

            #check accuracy over time (every 5 passes)
            if epoch % 5 == 0:
                training_cost, training_summery = session.run([cost, summary], feed_dict={X: training_data_x_scaled, Y: training_data_y_scaled})
                testing_cost, testing_summery = session.run([cost, summary], feed_dict={X: testing_data_x_scaled, Y: testing_data_y_scaled})
                print("Epoch {} - Training cost {} - Testing cost {}".format(epoch, training_cost, testing_cost))

                # log the summery
                training_log_writer.add_summary(training_summery, epoch)
                testing_log_writer.add_summary(testing_summery, epoch)

    #training complete or loaded from previously saved model
    print("Training is complete")
    final_training_cost = session.run(cost, feed_dict={X: training_data_x_scaled, Y: training_data_y_scaled})
    final_testing_cost = session.run(cost, feed_dict={X: testing_data_x_scaled, Y: testing_data_y_scaled})
    print("Final Training Cost {} ".format(final_training_cost), "Final Testing Cost {}".format(final_testing_cost))

    #Once training is complete, we can test it out
    #So, run the prediction operation on x testing data; this data is scaled
    predicted_sqrt_scaled = session.run(prediction, feed_dict={X: testing_data_x_scaled})

    #now unscale the data
    predicted_sqrt = y_scaler.inverse_transform(predicted_sqrt_scaled)
    for i in range(20):
        print("Squareroot of {} is {} in actual. The predicted squareroot is {}".format(testing_data_tf["actual_number"][i], testing_data_tf["actual_result"][i], predicted_sqrt[i]))


    #save the model
    save_path = saver.save(session, "./logs/sqrt_model.ckpt")
    print("Model was saved at {}".format(save_path))
