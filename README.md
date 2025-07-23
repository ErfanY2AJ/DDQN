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
