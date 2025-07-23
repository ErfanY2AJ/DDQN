# DDQN
Explain to how implement DQN and DDQN use as a classification Model

1. Sequential/Interactive Classification
Scenario: You must classify a data point, but you can take actions to gather more information before making a final decision. Each action (like “ask for another feature”, “request another test”, or “wait for another signal”) has a cost or reward.
Example: Medical diagnosis—an RL agent decides which test to run (actions) before giving a diagnosis (classification).

2. Learning with Delayed Rewards
Scenario: The reward for your classification is not immediate but comes after a sequence of actions.
Example: In a dialogue system, you may classify user intent at the end of a conversation. The RL agent gets a reward only if the whole conversation is successful.

3. Noisy or Weak Supervision
Scenario: You do not have ground-truth class labels for every data point, but you can observe a reward signal after making a classification. The RL framework learns a policy that maximizes expected reward.
Example: Online recommendation: Your RL agent “classifies” users into groups and recommends content. If the user interacts, you get a positive reward.

4. Interactive/Active Learning
Scenario: The agent actively chooses which sample to classify or which data to label next, receiving feedback or reward for accuracy and exploration.
Example: In active learning, you want to maximize model improvement per label requested.

5. Imitation Learning
Scenario: Instead of “predicting” a class label, the RL agent is trained to imitate an expert’s classification decisions as a policy.


# import libraries and parameters
````python
import numpy as np
import pandas as pd
from keras import layers
from keras import Input
from keras.models import Model
from keras.models import load_model
from keras import regularizers
from keras import optimizers
import random
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import OneHotEncoder
#from sklearn.model_selection import train_test_splits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from collections import deque
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import KBinsDiscretizer

#initialize random seeds so results are repeatable
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

DISCOUNT = 0.95
REPLAY_MEMORY_SIZE = 2000
MIN_REPLAY_MEMORY_SIZE = 512 #256
MIN_BATCH_SIZE = 128
EPISODES = 100
epsilon = 0.9 # 0.9
epsilon_decay = 0.99 #0.8
MIN_EPSILON = 0.0001
BATCH_SIZE=32
ALPHA=1
ALPHA_DECAY = 1 # 1 0.9 # 0.9999 #0.9975
ALPHA_MIN = 0.0001
# ssc = StandardScaler()
ssc=MinMaxScaler()
````

now we need to define our DQN as a Class notation this class should conclude :
       1- a nural network for estimate Q values that each action takes by defined Agent
       2-  How to play our Agent and How  to clarify your action's chain is going to solve my problem (reward and punishment)
       3- make a container to save what our agent dose like a smart video game (which means reply buffer)
       4- estimate the future Q (Q') with the initioation value 
       5- Guiding the selection of the optimal action at each step based on the Q-value, which leads to the transformation of DQN into DDQN.
       6- fit the model using labels arrarys Q_list and a batch_size (cause we do it offline)

# based on the explaination we wrote our python code from scratch 
``` python
class DQN():
    def __init__(self, input_size,output_size, data, labels):
        self.model = self.create_model(input_size, output_size)
        self.target_model = self.create_model(input_size, output_size)
        self.obs = data
        self.labels = labels
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        pass
    def create_model(self, input_size, output_size):

        input_tensor = Input(shape=(input_size,))
        x = layers.Dense(64,activation='relu')(input_tensor)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(16, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        output_tensor = layers.Dense(output_size, activation=None)(x)
        model = Model(inputs=input_tensor, outputs=output_tensor)
        #print (model.summary())
        ad = keras.optimizers.Adam(learning_rate=0.001) #'rmsprop'
        model.compile(optimizer=ad,loss='mse', metrics=['mae'])
        return model
    def play(self, obs, action):

        r = np.where((self.obs==obs).all(axis=1))
        if len (r[0]) == 0 :
            # something really wrong observation not in the game
            print("........... SOMETHING WRONG ................")
            return -100
        if self.labels[r[0][0]]  == action :
            return 100
        else :
            return -100
    def update_replay(self,obs,action,label, reward):
        self.replay_memory.append(((obs),action,label, reward))
    def get_qs(self, obs):
        return self.model.predict(obs, batch_size=1)
    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE :
            return
        minibatch = random.sample(self.replay_memory, MIN_BATCH_SIZE)
        # conevrt minibatch in list type to y numpy array. This is needed for access
        # also keras and TF except in numpy format
        y=np.array([np.array(xi, dtype=np.dtype(object)) for xi in minibatch])
        # extract features as a numpy array. As needed by Keras.
        # shape is (batch_size, feature_size)
        z=np.array([np.array(xi) for xi in y[:,0]])
        current_qa_list = dqn.model.predict(z, batch_size=BATCH_SIZE)
        future_qa_list = dqn.target_model.predict(z, batch_size=BATCH_SIZE)
        max_future_qa = np.argmax(future_qa_list, axis=1)
        alpha = ALPHA
        #qa = np.zeros(shape=(minibatch.shape[0],3))
        # Here only update the argmax because in theory we select max action
        # hence only the Q value of argmax get affected.
        for indx in range (y.shape[0]):
            #alpha1=1 gives 97 accuracy
            #current_qa_list[indx, y[indx, 1]] = (alpha1 * y[indx,3]) + ((1-alpha1)*max_future_qa[indx])
            #y[indx,1] is the action
            #y[indx,3] is the reward
            # z is the states / observation in numpy format
            #current_qa_list[indx, y[indx, 1]] = (alpha * y[indx,3]) + ((1-alpha)*max_future_qa[indx])
            current_qa_list[indx, y[indx, 1]] = current_qa_list[indx, y[indx, 1]] + (alpha * (y[indx,3])-current_qa_list[indx, y[indx, 1]] )
            pass
        dqn.model.fit(z, current_qa_list, epochs=16, batch_size=BATCH_SIZE, verbose=0)
        return

```

after that . we just need to prepare our data for training loop process , you can use the following code and you can change it based on your database shape , note that you dont need encoding labels cause we w'll do it on training phase 

``` python 
train_df = pd.read_csv('/content/train_dataset.csv')
test_df = pd.read_csv('/content/test_dataset.csv')

# 2. Split into X (features) and y (labels)
train_data = train_df.iloc[:, :-1].values
train_labels = train_df.iloc[:, -1].values
test_data = test_df.iloc[:, :-1].values
test_labels = test_df.iloc[:, -1].values 

train_data=ssc.fit_transform(train_data)
# train_data=digitizer.fit_transform(train_data)

test_data=ssc.fit_transform(test_data)
# test_data=digitizer.fit_transform(test_data)

train_labels=train_labels-1
test_labels=test_labels-1

dqn = DQN(train_data.shape[1], 4, train_data, train_labels)
```







