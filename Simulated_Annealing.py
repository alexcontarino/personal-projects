# Set Up
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import Counter
import pickle
from itertools import islice
import pandas as pd
import scipy.stats
import ast
import random
import math
import copy

## Import needed data sets. Stored on my Google Drive; links below to access.

# computed win probabilities for every seed matchup
winprobs_url = 'https://drive.google.com/file/d/1lcF2O3PdWYE6I4PJ-S51_8-DOU0suc5U/view?usp=sharing'
winprobs_path = 'https://drive.google.com/uc?export=download&id=' + winprobs_url.split('/')[-2]
win_probs_dirty = pd.read_csv(winprobs_path, index_col=0)

# dataframe describing possible seed matchups for first 3 rounds of tournament
matchups_url = 'https://drive.google.com/file/d/18WpgTMtClmzn_BcC6eiU41gEibjsBks1/view?usp=sharing'
matchups_path = 'https://drive.google.com/uc?export=download&id=' + matchups_url.split('/')[-2]
tournament_matchups_df = pd.read_csv(matchups_path, index_col=0)

## Tidying the win_probs_dirty data set

win_probs_dirty = win_probs_dirty.reset_index()
win_probs = pd.melt(win_probs_dirty, id_vars='index', value_vars=[str(i) for i in range(1,17)])
win_probs.rename(columns={'index': 'seed1',
                                'variable': 'seed2',
                                'value': 'seed1_win_prb'},
                        inplace=True)
win_probs.seed2 = win_probs.seed2.astype(int)
win_probs

## re-stores tournament_matchup_df as a new dictionary
## strings values in df converted to list items in dict

tournament_matchups = tournament_matchups_df.to_dict('index')

for seed in range(1,17):
  tournament_matchups[seed]['R2'] = list(ast.literal_eval(tournament_matchups[seed]['R2']))
  tournament_matchups[seed]['S16'] = list(ast.literal_eval(tournament_matchups[seed]['S16']))

"""# Step 2: Tournament Simulation"""

def region_simulation(tournament_matchups, win_probs, rng):

  wins_by_seed = {seed: 0 for seed in range(1,17)}

  ## First Round
  R1_winners = []
  for seed in range(1,9):
    opponent = tournament_matchups[seed]["R1"]
    seed1_win_prb = win_probs.loc[(win_probs.seed1==seed) & (win_probs.seed2==opponent), "seed1_win_prb"].item()
    if seed1_win_prb > rng.random():
      R1_winners.append(seed)
      wins_by_seed[seed] += 1
    else:
      R1_winners.append(opponent)
      wins_by_seed[opponent] += 1

  ## Round of 32
  already_matched = []
  R2_winners = []
  for seed in R1_winners:
    # find opponent
    if seed in already_matched:
      continue
    already_matched.append(seed)
    poss_opponents = tournament_matchups[seed]["R2"]
    for x in poss_opponents:
      if x in R1_winners:
        opponent = x
        already_matched.append(opponent)
        break
    seed1_win_prb = win_probs.loc[(win_probs.seed1==seed) & (win_probs.seed2==opponent), "seed1_win_prb"].item()
    if seed1_win_prb > rng.random():
      R2_winners.append(seed)
      wins_by_seed[seed] += 1
    else:
      R2_winners.append(opponent)
      wins_by_seed[opponent] += 1

  ## Sweet 16
  already_matched = []
  S16_winners = []
  for seed in R2_winners:
    # find opponent
    if seed in already_matched:
      continue
    already_matched.append(seed)
    poss_opponents = tournament_matchups[seed]["S16"]
    for x in poss_opponents:
      if x in R2_winners:
        opponent = x
        already_matched.append(opponent)
        break
    seed1_win_prb = win_probs.loc[(win_probs.seed1==seed) & (win_probs.seed2==opponent), "seed1_win_prb"].item()
    if seed1_win_prb > rng.random():
      S16_winners.append(seed)
      wins_by_seed[seed] += 1
    else:
      S16_winners.append(opponent)
      wins_by_seed[opponent] += 1

  ## Elite 8
  seed1 = S16_winners[0]
  seed2 = S16_winners[1]

  seed1_win_prb = win_probs.loc[(win_probs.seed1==seed) & (win_probs.seed2==opponent), "seed1_win_prb"].item()
  if seed1_win_prb > rng.random():
    region_winner = seed1
    wins_by_seed[seed1] += 1
  else:
    region_winner = seed2
    wins_by_seed[seed2] += 1

  return region_winner, wins_by_seed

def tournament_simulation(tournament_matchups, win_probs, rng):

  region_winners = []
  region_results = {}
  all_results = {seed: 0 for seed in range(1,17)}

  for region in range(4):
    #print(rng)
    region_seed = rng.integers(99999999999999)
    region_rng = np.random.default_rng(seed = region_seed)
    region_winner, region_wins_by_seed = region_simulation(tournament_matchups, win_probs, region_rng)
    region_results[region] = region_wins_by_seed
    region_winners.append(region_winner)
    all_results = {seed: (all_results[seed] + region_wins_by_seed[seed]) for seed in range(1,17)}

  #Final Four
  F4_winners = []
  champ_regions = []

  for (i,j) in [(0,1),(2,3)]:
    seed1, seed2 = region_winners[i], region_winners[j]
    seed1_win_prb = win_probs.loc[(win_probs.seed1==seed1) & (win_probs.seed2==seed2), "seed1_win_prb"].item()
    if seed1_win_prb > rng.random():
      F4_winners.append(seed1)
      all_results[seed1] += 1
      region_results[i][seed1] += 1
      champ_regions.append(i)
    else:
      F4_winners.append(seed2)
      all_results[seed2] += 1
      region_results[j][seed1] += 1
      champ_regions.append(j)

  #Championship
  seed1 = F4_winners[0]
  seed1_region = champ_regions[0]
  seed2 = F4_winners[1]
  seed2_region = champ_regions[1]

  champion = None

  seed1_win_prb = win_probs.loc[(win_probs.seed1==seed1) & (win_probs.seed2==seed2), "seed1_win_prb"].item()
  if seed1_win_prb > rng.random():
    champion = seed1
    all_results[seed1] += 1
    region_results[seed1_region][seed1] += 1
  else:
    champion = seed2
    all_results[seed2] += 1
    region_results[seed1_region][seed2] += 1

  sim_results = {"champion":champion,
                 "all_results": all_results,
                 "region_results": region_results}

  return sim_results

# test cell
_, wins_by_seed = region_simulation(tournament_matchups, win_probs, np.random.default_rng(seed=69))

wins_by_seed

"""# Step 3: Helper Functions


"""

def win_calculator(ballot, omega, rng):

  # This is your objective function that depends on the decision variable x and random seed
  # x = dictionary of teams bought by seed
  # omega = dictionary of tournament results

  region_results = omega["region_results"]
  all_results = omega["all_results"]

  wins = 0

  for seed in range(1,17):
    num_seeds_bought = int(ballot[seed])
    if num_seeds_bought == 0:
      continue
    elif num_seeds_bought == 4:
      wins += all_results[seed]
    else:
      regions = rng.choice([0,1,2,3], num_seeds_bought)
      for region in regions:
        wins += region_results[region][seed]

  return wins

def under_cost(ballot, costs_dict):
    """Constraint function: cost. Returns True if ballot is cost feasible."""
    # Cost sum of ballot must be less than $100

    cost = 0

    for seed in range(1,17):
      cost += costs_dict[seed] * ballot[seed]

    return cost <= 100

def neighbor_function(x, rng, costs_dict):
    """Generates a neighbor solution within search space:
      Integers between 0 and 4 (inclusive) for each seed s.t. sum of cost <= 100
      x = ballot
      rng = provide rng
    """


    valid_cost = False
    while not valid_cost:
        candidate = x
        for seed in candidate:
          if candidate[seed] == 0:
            candidate[seed] += rng.choice([0,1])
          elif candidate[seed] == 4:
            candidate[seed] += rng.choice([-1,0])
          else:
            candidate[seed] += rng.choice([-1,0,1])
        valid_cost = under_cost(candidate, costs_dict)       # Ensure the candidate is within cost

    return candidate

# Define a function to calculate the k-th percentile
def objective_function(ballot, N, rng, win_threshold=18):

  global tournament_matchups, win_probs

  # Generate N samples for the stochastic objective function
  samples = []
  for _ in range(N):
    omega = tournament_simulation(tournament_matchups, win_probs, rng) # Generate random omega (sampling from distribution)
    samples.append(win_calculator(ballot, omega, rng))  # Evaluate the objective

  # Calculate the desired percentile of the samples
  prb_success = np.sum(np.array(samples) >= win_threshold) / N

  return prb_success

"""# Step 4: Main Algorithm"""

def simulated_annealing(initial_solution, initial_temp, cooling_rate, max_iterations, N, percentile, rng):
    """Simulated Annealing with constrained search space."""

    # Set up cost_dict
    costs_dict = {1: 25,
         2: 19,
         3: 13,
         4: 12,
         5: 11,
         6: 10,
         7: 8,
         8: 5,
         9: 5,
         10: 4,
         11: 4,
         12: 3,
         13: 2,
         14: 2,
         15: 1,
         16: 1}

    current_solution = initial_solution
    current_value = objective_function(current_solution, N, rng, percentile)
    print(f"Initial Solution = {current_solution}, Initial Value = {current_value}")
    temperature = initial_temp

    best_solution = current_solution
    best_value = current_value

    for iteration in range(max_iterations):
        print(f"\nITERATION {iteration+1}...")
        # Generate a neighboring solution
        candidate_solution = neighbor_function(current_solution, rng, costs_dict)
        candidate_value = objective_function(current_solution, N, rng, percentile)

        # Calculate acceptance probability
        delta = candidate_value - current_value
        if delta > 0:
            print(f"Candidate value better than current value")
            acceptance_probability = 1.0
        else:
            acceptance_probability = np.exp(delta / temperature)
            print(f"Accepting Worse Solution With Probability {100*acceptance_probability:.2f}%")

        # Accept the candidate solution with a certain probability
        if rng.random() < acceptance_probability:
            print(f"Candidate accepted")
            current_solution = candidate_solution
            current_value = candidate_value

            # Update the best solution found so far
            if current_value > best_value:
                best_solution = current_solution.copy()
                print("best solution has changed")
                best_value = current_value
                print(f"New Best Solution = {best_solution}, Best Value = {best_value}")
            else:
                print("candidate solution chosen, but it is not the best solution")
                print(f"New current value: {current_value}")
        else:
            print("Candidate rejected")

        # Cool down the temperature
        temperature *= cooling_rate

        # Optionally, print progress
        if (iteration+1) % 10 == 0:
            print(f"Iteration {iteration+1}: Best Solution = {best_solution}, Best Value = {best_value}")

        # Stop once temperature is very low
        if temperature < 1e-6:
            break

    return best_solution, best_value

"""# Code Execution"""

# Parameters
rng = np.random.default_rng(seed=42)
initial_solution = {1: 4, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0}
initial_temp = 100
cooling_rate = 0.99
N = 2500
max_iterations = 1000
win_threshold = 18

# Run the algorithm
best_solution, best_value = simulated_annealing(
    initial_solution, initial_temp, cooling_rate, max_iterations, N, win_threshold, rng
)

print(f"\Beste Solution: {best_solution}")
print(f"Best Value: {best_value}")