#!/bin/bash

# This script runs the evaluation of the DQN agent on the game instances of the human subject study and saves the results to a csv file.

source env/bin/activate

human_policy='greedy' # Here by human we refer to the agent that takes the actions.
human_type='dqn' # Type of the score function used by the agent that takes actions.
agent_type='dqn' # Type of the AI agent used by the decision support policy.
arm=1.0 # Arm value (the action sets include all the actions).
# The above configuration equals to the DQN agent acting alone.
python -m src.evaluate_agent  --games_from_file --human_policy $human_policy --human_type $human_type --agent_type $agent_type --arm $arm