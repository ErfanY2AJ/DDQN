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


first we need import libraries and parameter used on Q-learning
````
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
