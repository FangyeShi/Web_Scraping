import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# This is a script to train a sentiment classifier using a pretrained word-embedding model and a NN with 2 hidden layers.
# See tutorial: https://www.tensorflow.org/hub/tutorials/text_classification_with_tf_hub

# Read dataset
long_clean = pd.read_csv('long.csv',index_col=0)
short_clean = pd.read_csv('short.csv',index_col=0)

# Change all words to lower_case.
long_clean['title'] = long_clean['title'].apply(lambda t: " ".join(word.lower() for word in t.strip().split()))
# Remove noncharacter/nonspace.
long_clean['title'] = long_clean['title'].apply(lambda t: re.sub('[^\w\s]','', t))
# Remove extra spaces.
long_clean['title'] = long_clean['title'].apply(lambda t: re.sub('\s+',' ', t.strip()))
# Now each title will be a string of words separated by exactly one space.

# Replace np.nan by empty string.
long_clean['summary'] = long_clean['summary'].apply(lambda t: '' if pd.isnull(t) else t)
# Change all words to lower_case.
long_clean['summary'] = long_clean['summary'].apply(lambda t: " ".join(word.lower() for word in t.strip().split()))
# Remove noncharacter/nonspace.
long_clean['summary'] = long_clean['summary'].apply(lambda t: re.sub('[^\w\s]','', t))
# Remove extra spaces.
long_clean['summary'] = long_clean['summary'].apply(lambda t: re.sub('\s+',' ', t.strip()))
# Now each summary will be a string of words separated by exactly one space.

# Do the same thing for short ideas:
short_clean['title'] = short_clean['title'].apply(lambda t: " ".join(word.lower() for word in t.strip().split())) # change to lower_case
short_clean['title'] = short_clean['title'].apply(lambda t: re.sub('[^\w\s]','', t)) # remove punctuations
short_clean['title'] = short_clean['title'].apply(lambda t: re.sub('\s+',' ', t.strip()))
short_clean['summary'] = short_clean['summary'].apply(lambda t: '' if pd.isnull(t) else t)
short_clean['summary'] = short_clean['summary'].apply(lambda t: " ".join(word.lower() for word in t.strip().split())) # change to lower_case
short_clean['summary'] = short_clean['summary'].apply(lambda t: re.sub('[^\w\s]','', t)) # remove punctuations
short_clean['summary'] = short_clean['summary'].apply(lambda t: re.sub('\s+',' ', t.strip()))

# Create dataframe consists of text = title+summary and add polarity: long =1 , short =0
X = pd.DataFrame({'text':long_clean['title']+' '+long_clean['summary']})
X['polarity']=1
Y = pd.DataFrame({'text': short_clean['title']+' '+short_clean['summary']})
Y['polarity']=0

# Split into training and testing set with a 8:2 ratio
X_train, X_test= train_test_split(X, test_size = .2)
Y_train, Y_test= train_test_split(Y, test_size = .2)

# Concat train set and randomly suffle the entries and reset the index
train_df = pd.concat([X_train, Y_train]).sample(frac=1).reset_index(drop=True)
test_df = pd.concat([X_test,Y_test]).reset_index(drop=True)

# Training input on the whole training set with no limit on training epochs.
train_input_fn = tf.estimator.inputs.pandas_input_fn(train_df, train_df["polarity"], num_epochs=None, shuffle=True)

# Prediction on the whole training set.
predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(train_df, train_df["polarity"], shuffle=False)
# Prediction on the test set.
predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(test_df, test_df["polarity"], shuffle=False)

# A helper function to get predictions
def get_predictions(estimator, input_fn):
  return [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]

# A helper function to train models with different options
def train_and_evaluate_with_module(hub_module, train_module=False):
    '''
    hub_module is some pretrained module provided by tf-hub
    train_module is a boolean variable indicating wether we want to train the pretrained module together with the NN

    Will store the confusion matrix as .png image in current direct

    Return a dictionary of training accuracy and test accuracy
    '''
    # The pretrained model provide a text embedding.
    # First, it will split the text into words. Then it will embed each word into a 128 dimensional space and finally it takes
    # the sum of the word vectors divided by square-root of the length of the text as the embedding of the text.
    embedded_text_feature_column = hub.text_embedding_column(key="text", module_spec=hub_module, trainable=train_module)

    # Train a NN classifier after embedding.
    # There are parameters that could be tuned later.
    # Roughly 114k parameters in NN
    estimator = tf.estimator.DNNClassifier(hidden_units=[500, 100],feature_columns=[embedded_text_feature_column],n_classes=2,optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

    # Train 1000 steps. Note the default batch size is 128. With a training set of size ~ 15000, this is roughly 8.5 epochs.
    estimator.train(input_fn=train_input_fn, steps=1000)

    # Evaluate the model
    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

    training_set_accuracy = train_eval_result["accuracy"]
    test_set_accuracy = test_eval_result["accuracy"]

    print('='*50)
    print("Training finished!")
    print("Training set accuracy: {}".format(training_set_accuracy))
    print("Test set accuracy: {}".format(test_set_accuracy))
    print('='*50)


    # Plot the confusion matrix and save to the same directory
    LABELS = ["negative", "positive"]

    # Create a confusion matrix on training data.
    with tf.Graph().as_default():
      cm = tf.confusion_matrix(train_df["polarity"],get_predictions(estimator, predict_train_input_fn))
      with tf.Session() as session:
        cm_out = session.run(cm)

    # Normalize the confusion matrix so that each row sums to 1. This makes the plot easy to read.
    cm_out = cm_out.astype(float) / cm_out.sum(axis=1)[:, np.newaxis]

    # clear previous plot
    plt.clf()
    plt.cla()
    plt.close()
    # Plot new graph
    sns.heatmap(cm_out, annot=True, xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Get module name to be used in the name of .png file.
    module_name = hub_module.replace('//','/')
    module_name = module_name.split('/')[-2]
    if train_module:
        plt.savefig(module_name+'-with-module-training.png')
    else:
        plt.savefig(module_name+'.png')

    return {"Training accuracy": training_set_accuracy,"Test accuracy": test_set_accuracy}



results = {}
results["nnlm-en-dim128"] = train_and_evaluate_with_module("https://tfhub.dev/google/nnlm-en-dim128/1")
results["nnlm-en-dim128-with-module-training"] = train_and_evaluate_with_module("https://tfhub.dev/google/nnlm-en-dim128/1", True)
results["random-nnlm-en-dim128"] = train_and_evaluate_with_module("https://tfhub.dev/google/random-nnlm-en-dim128/1")
results["random-nnlm-en-dim128-with-module-training"] = train_and_evaluate_with_module("https://tfhub.dev/google/random-nnlm-en-dim128/1", True)

# Total train time is roughly 3 minutes for all 4 models using gpu GTX 1060.

pd.DataFrame.from_dict(results, orient="index").to_csv('model_accuracies.csv')
