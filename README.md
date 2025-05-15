# March Madness Ballot Game: Different Analytic Approaches

## Introduction
This notebook executes an evolutionary algorithm to find an optimal ballot for the March Madness ballot game developed by Dr James Stapleton (MSU Statistics Department). The game is a knapsack-type optimization problem: create a ballot of teams that maximizes wins while remaining within a specified budget (typically 100 points or imaginary dollars), with higher seeded teams being more expensive. Number 1 seeds typically cost 25, 2 seeds 19, 3 seeds 13, and so on down to the 15th and 16th seeds which cost 1. Additionally, win is weighted the same regardless of round (e.g., winning a game in the first round is just as much of a win as winning the Champsionship game). 

A winning ballot depends on the year but will generally accumulate 18-20 wins. The fundamental problem is whether to chose a few higher seeded teams, many low seeded teams, or some mixture in between, in order to maximize the probability of a winning score.

## Table of Contents

1. [Defining the Problem](#defining-the-problem)
1. [Integer Programming Approach](##integer-programming-approach)
1. [Simulated Annealing](#simulated-annealing)
1. [Genetic Algorithm](#genetic-algorithm)
1. [Results and Findings](#results-and-findings)
1. [A Note on Seed Win Probabilities](#a-note-on-win-probabilities)
1. [Acknowledgments](#acknowledgments)

## Defining the Problem

We can describe the problem as:

Maximize the probability, $p$, of achieving a winning score, $M$, from a ballot chosen of teams $j$ in $ {1, 2, 3, 4} $ seeded $i$ from $ {1, 2, ..., 16} $ : $$ P \left( \sum{W_{ij}*1_{ij}} \geq M \right) $$

subject to the constraints that:

- No more than four teams of any seed can be bought: $$ \sum_{j}1_{ij} \leq 4 \space\space \forall i \in \{1, 2, ..., 16 \} $$   
- The cost of the ballot is less than the allowable budget, typically 100: $$ \sum_{i,j}C_{i}*1_{ij} \leq 100 $$

Where:
- $W_{ij}$ is the number of wins achieved by team $j$ seeded $i$
- $1_{ij}$ is an indicator variable for that team being chosen for the ballot
- $C_{ij}$ is its cost to buy, and
- $M$ is the assumed winning score (I chose 18 for this project).

### Motivation

What's fun about this optimization problem is the nature of the objective function: it cannot be directly observed given that each $W_{ij}$ is random. Rather, the objective function must somehow be estimated. This opens up a lot of potential means to try to "solve" this game, each with their own trade-offs.

To simplest approach to determining the optimal ballot is to summarize the distributions of $W_{ij}$ by seed, say by computing their mean or median using historical data, allowing for an integer programming apprach. Details on this implementations are shared below.

This approach, however, abstracts away from the the tournament structure and the probabilistic nature of winning the game. On the other end of the spectrum, we could simulate the outcomes of $W_{ij}$ for each team in the tournament, requiring precise team data. But determining team strengths, how those translate to win probabilities, and so on, are substantial projects in their own right.

One middle ground is to simulate the tournament at the seed level. Using previous tournament results to generalize relative team strengths based solely upon seed match-ups, simple tournament simulations can be instituted to account for tournament structure and to model the distribution of outcomes for a given ballot. In such a set-up, an efficient search heuristic is needed to navigate through the large set of potential solutions. I share details of my implementation of two approaches: a basic simulated annealing algorithm, and a genetic algorithm with my own tweaks that (I think) help apply the basic algorithm to this problem set.

## Integer Programming Approach

Details coming soon.

## Simulated Annealing

I implement a simulated annealing algorithm with the following hyperparameters to search the space of potential ballots ("candidates"): 

- Initial Temperature: 100
- Cooling Rate: 0.99
- Maximum Iterations: 1000 

The initial solution given to the algorithm is the candidate which spends all money on the 1 seeds (4x teams worth 25 each) and its win probability is estimated. At each iteration, a neighbor function generates a candidate whose ballot is random perturbations of the current candidate. This new candidate is evaluated by simulating the tournament 2,000 times and counting the number of times the candidate meets or exceeds the win threshold (18 wins) to arrive at an estimate of its win probability. From there, the usual steps of a simulated annealing algorithm are applied:

1. Accept the new candidate if its win probability exceed the previous best. Or, if it is not the new best solution, accept it with probability $q$, which is a function of the difference to the best win probability and the temperature of the system. 
1. If the previous candidate is reject, return to previous candidate. 
1. Update temperature according to cooling rate. 

Repeat until the maximum number of iterations is reached. At this point, the temperature of the system is 0.004. 

## Genetic Algorithm  

To search the space of possible ballots, I implement a genetic algorithm which evaluates potential ballots ("candidates") by estimating their winning score probability $p$ across generations through Bayesian updating, where the conjugate prior is the Beta distribution.

The algorithm is as follows:

**Step 1**: Generate a pool of $C$ initial candidates (set to 100). Their probabliity of obtaining a winning score $p$ is assumed to follow a Beta distribution. The hyperparameters $\alpha$ and $\beta$ are determined for the first generation of candidates by a baseline guess that a random ballot achieves a winning score (set to 0.025) and a tunable strength parameter. For this project I set strength to be 10, meaning that $\alpha$ and $\beta$ add up to 10 and satisfy an initial guess $p$ of 0.025. The initial guess and strength of this prior are in order to be pessimistic when evaluating new candidates but overcome in short order if a candidate shows promise (surviving 2 or 3 generations in the pool).

**Step 2**: Simulate the March Madness tournament $n$ times (set to 30) for each candidate in the pool. The hyperparameters $\alpha$ and $\beta$ are then updated for each candidate by counting the number of times a winning score ($\geq18$) is reached out of the $n$ new trials. 

**Step 3**: Generate $k$ number of "children" candidates as neighbors to their parents (set to 5). The hyperparameters for children candidates are initialized using a weighted combination of the baseline guess (0.025) and their parent's current guess, which I call a "regress factor." The strength of the prior is the same as set initially for all candidates (10). The intended effect is for the children's estimates to be influenced by their parents but with some amount of "regression to the mean" and with the strength of the priors being equal for all new candidates. I set this regression to be fairly small in the experiment--0.05 (5% baseline guess; 95% parent's current estimate)--to encourage exploration.

**Step 4**: The priors for the children candidates are updated by again simulating the tournament $n$ time for each (i.e., run **Step 2** for each child).

**Step 5**: Rank all candidates in descending order by the computed lower confidence bound of their winning score probability using their $\alpha$ and $\beta$ values.

**Step 6**: Select the top $C$ candidates to move to the next generation.

Repeat Steps 2-6 until... some stopping criteria is reached. This is where I still need to think through the algorithm more. For this experiment, I set this merely as reaching the maximum number of iterations (set to 100). Because I  encourage exploration by setting a small regress factor, candidate rankings are fluid as promising new candidates show promise and then (more often than not) fade. Without a clear stopping strategy, I choose the best candidate by evaluating the top 10 ranked candidates 20,000 times to arrive at very precise estimates of their win probability. 

## Results and Findings

### Integer Programming Approach

Results coming soon. 

### Simulated Annealing 

After 100 iterations of the genetic algorithm, the best ballot obtained an estimated win probability of ____. The suggested ballot is as follows:

- 3 seeds: __ teams bought
- 4 seed: __
- 5 seed: __
- 8 seed: __
- 10 seed: __
- 12 seed: __
- 15 seed: __

Using the simulated annealing algorithm here does feel a bit like shooting in the dark: in a discrete space, the objective function is not smooth and it is difficult to identify which regions are promising. Additionally, limits of computation time (I tried to keep things less than 24 hours) meant each candidate having 2,000 tournament simulations at its disposal to estimate its win probability. With that probability appearing to be around 0.25, the margin of error on 2,000 trials is roughly 0.02, or 10%! 

These concerns were the main motivations for pursuing the genetic algorithm approach.  

### Genetic Algorithm

After 100 iterations of the genetic algorithm, the best ballot obtained an estimated win probability of ____. The suggested ballot is as follows:

- 3 seeds: 3 teams bought
- 4 seed: 1
- 5 seed: 2
- 8 seed: 3
- 10 seed: 1
- 12 seed: 2
- 15 seed: 2

### Key Takeaways

1. The best performing ballots achieve around a 28% win probability. They typically have around 15 teams, with the majority of the budget spent on high-value seeds (seeds 3, 8, 10), and the remainder rounded out with as many low-cost teams as the player can still afford. 
1. 1 and 2 seeds are not worth their respective price tags.
1. While each algorithm identifies a top candidate, based on the results from the integer programming approach and the genetic algorithm, there are many ballots that provide similar win probability based on roughly equivalent combinations of seeds bought.
1. Recognizing the year-to-year variance of seed quality within the tournament, and considering the above observations, a good rule of thumb for constructing a ballot is:

  - Spend ~60% of the budget on seeds 3 - 8
  - Spend ~30% of the budget on seeds 9 - 12
  - Spend the remainder on seeds 13 - 16.

## A Note on Win Probabilities  

First, win probabilities for each potential seed matchup are roughly estimated based on a combination of historical data and assumptions. For logical consistency, I assume that the probability a team seed $i$ defeats a team seed $j$ is:
1. $0.5$ if $i = j$,
2. $>0.5$ if $i < j$, and
3. $< 0.5$  if $i > j$.

Historical data from previous tournaments indicates that seed $1$ defeats seed $16$ with a probability of ~0.99. The win probabilities for every other combination of seeds playing one another is a linear interpolation from this result and the logic rules above. There is likely some non-linear behavior in the plane of win probabilities across the combinations of match-ups. My hunch is that the difference in quality between teams seeded 1 versus 2 is greater than those seeded, say, 8 versus 9, and should be reflected in the win probabilities. Better estimates of win probabilities will be a future refinement to the project (hopefully in-time for next year's tournament).

## Acknowledgements

Some of code below is AI-generated (ChatGPT and Google AI code assist).
