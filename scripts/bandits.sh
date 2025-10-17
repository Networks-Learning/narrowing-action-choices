#!/bin/bash

# This script runs the bandit algorithms on the human study data and saves the results to a csv file.

source env/bin/activate

L=150 # Lipschitz constant
beta=2 # Exploration parameter
gamma=0.99 # Discount factor
n=100 # Number of algorithm iterations per budget value

python -m src.bandits --L $L --beta $beta --gamma $gamma --n $n