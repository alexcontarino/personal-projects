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

"""# Tournament Simulation"""

def region_simulation(tournament_matchups, win_probs, rng):
  '''
  INPUTS:
  tournament_matchups: dictionary of potential matchups
  win_probs: dataframe of win probabilities for every seed matchup
  rng: random number generator
  OUTPUTS:
  region_winner: seed of region winner
  wins_by_seed: dictionary of number of wins by seed
  '''
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
  '''
  INPUTS:
  tournament_matchups: dictionary of potential matchups
  win_probs: dataframe of win probabilities for every seed matchup
  rng: random number generator
  OUTPUTS:
  sim_results: dictionary of simulation results:
  - seed of champion
  - dictionary of wins by seed for tournament
  - dictionary of results for each region (for debugging)
  '''

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

test_ballot = {
 1: 0.0,
 2: 0.0,
 3: 3.0,
 4: 0.0,
 5: 0.0,
 6: 0.0,
 7: 0.0,
 8: 3.0,
 9: 0.0,
 10: 4.0,
 11: 4.0,
 12: 4.0,
 13: 1.0,
 14: 0.0,
 15: 0.0,
 16: 0.0}

rng = np.random.default_rng(seed = 42)
sim_results = tournament_simulation(tournament_matchups, win_probs, rng)

"""# Helper Functions"""

def win_calculator(ballot, omega, rng):
  '''
  INPUTS:
  ballot: ballot of seeds bought
  omega: results of tournament simulation
  rng: random number generator
  OUTPUTS:
  wins: number of wins for ballot
  '''
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

def objective_function(ballot, N, rng, winning_score=18):
  '''
  INPUTS:
  ballot: ballot of seeds bought
  N: number of simulations to run
  rng: random number generator
  OUTPUTS:
  winning_sims: number of sims matching or exceeding the assumed winning score
  '''

  global tournament_matchups, win_probs

  # Generate N samples for the stochastic objective function
  samples = []
  for _ in range(N):
    omega = tournament_simulation(tournament_matchups, win_probs, rng) # Generate random omega (sampling from distribution)
    samples.append(win_calculator(ballot, omega, rng))  # Evaluate the objective

  # Calculate the number of sims matching or exceeding the assumed winning score
  winning_sims = np.sum(np.array(samples) >= winning_score)

  return winning_sims

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
        # rng.shuffle(movements)
        for seed in candidate:
          if candidate[seed] == 0:
            candidate[seed] += rng.choice([0,1])
          elif candidate[seed] == 4:
            candidate[seed] += rng.choice([-1,0])
          else:
            candidate[seed] += rng.choice([-1,0,1])
        valid_cost = under_cost(candidate, costs_dict)       # Ensure the candidate is within cost

    return candidate

def generate_candidate_with_rejection(num_candidates, costs_dict, rng):
    '''Generates intial set of candidates'''

    candidates = []
    while len(candidates) < num_candidates:
        # Generate a random solution (example: integer values between 0 and 10)
        candidate = {seed: rng.integers(0, 4) for seed in range(1,17)}

        # Check if the candidate satisfies constraints
        if under_cost(candidate, costs_dict):
            candidates.append(candidate)

    candidates = {i: candidates[i] for i in range(num_candidates)}

    return candidates

"""# Main Algorithm"""

def my_algo(game_parameters, algorithm_parameters, bayesian_parameters, rng, initial_candidates=None):
  '''
  INPUTS:
  game_paramenters: dictionary of game parameters (tournament matchups, win probabliities, costs)
  algorithm_parameters: dictionary of parameters to execute algorithm (number of candidates, number of simulations, stop criteria, etc.)
  bayesian_parameters: dictionary of parameters for Bayesian updating
  rng: random number generator
  initial_candidates (optional): import external set of candidates in case you pause experiment
  OUTPUTS:
  candidates: dictionary of candidates
  LCBs: dictionary of lower confidence bounds for each candidate
  params: dictionary of parameters for each candidate
  '''
  tournament_matchups = game_parameters["tournament_matchups"]
  win_probs = game_parameters["win_probs"]
  costs_dict = game_parameters["costs_dict"]

  c = algorithm_parameters["c"]
  n = algorithm_parameters["n"]
  k = algorithm_parameters["k"]
  max_iters = algorithm_parameters["max_iters"]

  initial_guess = bayesian_parameters["initial_guess"]
  strength_factor = bayesian_parameters["strength_factor"]
  regress_factor = bayesian_parameters["regress_factor"]

  alpha = strength_factor * (initial_guess)
  beta = strength_factor * (1 - initial_guess)

  if initial_candidates is not None:
    # import candidates if they are there
    candidates = initial_candidates["candidates"]
    params = initial_candidates["params"]
    LCBs = initial_candidates["LCBs"]
    best_LCB = np.max(list(initial_candidates['LCBs'].values()))
  else:
    # initialize c candidates and other variables
    candidates = generate_candidate_with_rejection(num_candidates = c, costs_dict = costs_dict, rng = rng)
    params = {key: [alpha, beta] for key in candidates.keys()}
    LCBs = {key: scipy.stats.beta.ppf(0.15, params[key][0], params[key][1]) for key in candidates.keys()}
    best_LCB = 0

  # for iter in max iter
  for iter in range(max_iters):

    iter_candidates = {}
    iter_params = {}
    iter_LCBs = {}

    # for each candidate
    for candidate, parent in candidates.items():

      iter_candidates[candidate] = copy.deepcopy(parent)
      iter_params[candidate] = copy.deepcopy([params[candidate][0], params[candidate][1]])
      iter_LCBs[candidate] = -1

      # evaluate candidate n times
      successes = objective_function(parent, n, rng)

      # update prior
      iter_params[candidate][0] += successes
      iter_params[candidate][1] += (n - successes)

      # compute LCB
      iter_LCBs[candidate] = scipy.stats.beta.ppf(0.15,
                                                  a=iter_params[candidate][0],
                                                  b=iter_params[candidate][1])

      # generate k neighbor "children", evaluate them, and compute their p_hats and LCBs
      parent_alpha = copy.deepcopy(iter_params[candidate][0])
      parent_beta = copy.deepcopy(iter_params[candidate][1])
      for _ in range(k):
        # initialize child and priors
        child = neighbor_function(parent, rng, costs_dict)
        for i in range(100000):
          if f"{candidate}.{i}" not in candidates:
            child_name = f"{candidate}.{i}"
            iter_candidates[child_name] = child
            break
        # priors are sliding scale between parent's priors and initial priors based on regress factor
        child_guess = regress_factor * (initial_guess) + (1 - regress_factor) * (parent_alpha / (parent_alpha + parent_beta))
        #child_alpha = regress_factor * (alpha) + (1 - regress_factor) * parent_alpha
        #child_beta = regress_factor * (beta) + (1 - regress_factor) * parent_beta
        child_alpha = strength_factor * (child_guess)
        child_beta = strength_factor * (1 - child_guess)
        iter_params[child_name] = [child_alpha, child_beta]

        # simulate, update, evaluate -->
        # evaluate candidate n times
        successes = objective_function(child, n, rng)
        # update priors
        iter_params[child_name][0] += successes
        iter_params[child_name][1] += (n - successes)
        # compute LCB
        iter_LCBs[child_name] = scipy.stats.beta.ppf(0.15,
                                                     a=iter_params[child_name][0],
                                                     b=iter_params[child_name][1])

    # rank order candidates and their children by LCB
    iter_LCBs_sorted = dict(sorted(iter_LCBs.items(), key=lambda item: item[1], reverse=True))

    # reassign previous values in prep for next iteration
    previous_candidates = copy.deepcopy(candidates)
    previous_best_LCB = best_LCB

    # Keep top c candidates to initialize data for next iteration
    LCBs = copy.deepcopy(dict(islice(iter_LCBs_sorted.items(), c)))
    candidates = copy.deepcopy({key: iter_candidates[key] for key in LCBs.keys()})
    params = copy.deepcopy({key: iter_params[key] for key in LCBs.keys()})

    # Print progress and results
    best_candidate = list(LCBs.items())[0][0]
    best_LCB = list(LCBs.items())[0][1]

    p_hat = params[best_candidate][0] / (params[best_candidate][0] + params[best_candidate][1])

    num_survived = len(set(candidates.keys()).intersection(previous_candidates.keys()))
    turnover = c - num_survived

    print(f"Iteration {iter+1} complete!")
    print(f"----Candidate {best_candidate} has a LCB of {best_LCB:.4f} and estimated probability of success of {p_hat:.4f}")
    print(f"----Best LCB increased by {(best_LCB - previous_best_LCB):.4f}")
    print(f"----{turnover} candidates replaced")

  return candidates, LCBs, params

"""# Code Execution"""

# Parameters
rng = np.random.default_rng(42)

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

game_parameters = {"tournament_matchups": tournament_matchups,
                   "win_probs": win_probs,
                   "costs_dict": costs_dict}

# ~10 min/iter to execute with these parameters on Google Colab CPU
algorithm_parameters = {"c": 100,                 # number of candidates to start each generation
                        "n": 30,                  # number of simulations to run for each candidate
                        "k": 5,                   # number of children by each candidate in each generation
                        "max_iters": 100}         # max number of iterations

bayesian_parameters = {"initial_guess": 1/40,     # baseline guess for the win_probability for each ballot (= alpha / (alpha + beta))
                       "strength_factor": 10,     # strength of prior (bigger = stronger prior) (= alpha + beta)
                       "regress_factor": 0.05}    # sliding scale between 0 and to determine prior for children:
                                                  #   0 --> child's guess = parent's current estimate
                                                  #   1 --> child's guess = initial guess
                                                  #   (strength of child prior is set by strength_factor)


## Run the algorithm
candidates, LCBs, params = my_algo(game_parameters, algorithm_parameters, bayesian_parameters, rng)

### Save results ##
algo_results = {"candidates": candidates,
                "LCBs": LCBs,
                "params": params}

# with open('_____INSERT_FILE_PATH_______.pkl', 'wb') as f:
#  pickle.dump(algo_results, f)

# obtain precise estimate for top-10 ranked candidates
iter = 0
n = 20000
for key, value in candidates.items():
  iter += 1
  p_hat = objective_function(candidates[key], n, rng) / n
  print(f"Candidate {key}: p_hat = {p_hat}")
  if iter == 10: break