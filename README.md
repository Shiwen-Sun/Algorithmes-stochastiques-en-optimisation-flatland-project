# Algorithmes-stochastiques-en-optimisation-flatland-project

The Flatland environment is a 2D gridworld for developing Multi-Agent Reinforcement Learning. Through simulating train networks as agents, this platform allows the development of policies for the efficient management of dense traffic on complex railway networks. We use this platform in a project on neuroevolution.

In this project, we evolve policies for each train in the environment, either using the same genes for each agent or different genes. Policy representation, stochastic algorithm, and agent modelling decisions such as inter-agent communication are all up to the students. Using a fixed budget of evaluations, we submit our results and the corresponding code for final evaluation.



Tasks and evaluation
We will focus on the evolution of solutions for three fixed environments: small.pkl, medium.pkl, and large.pkl. We have a budget of 1000 evaluations equivalent to the 1+lambda ES for 100 generations with a population size of 10. For the small environment, this should take around 10 minutes on an ordinary laptop.

Our final evaluation will be on the large.pkl environment, however this takes much longer to run so we are advised to develop solutions on the small.pkl environment first.

Oue task is to improve the performance of evolution of a solution. We explore whichever direction you think is most promising, including but not limited to:

- replacing the 1+lambda ES with a different algorithm
- changing evolutionary hyperparameters like population size and standard deviation
- modifying the agent observations
- modifying the policy network
- changing the fitness function

Our final results on the large.pkl environment will be evaluated using three criteria:

- final result during evolution (fitness at the end of an episode)
- convergence speed (how quickly the evolution improves)
- robustness to different random environments (as in test.py)



Afin d’optimiser la recherche, nous avons mis en place plusieurs algorithmes d’optimisation stochastique et comparé leurs performances : 

1. Descente du gradient (approx_gradient) qui donne un résultat légèrement meilleur sur small que 1+lamba : f_best = -650

2. CMA-ES (cma_optim) qui est efficace, mais il est difficile de trouver les meilleurs paramètres. 
Les résultats semblent cependant corrects pour std = 0.2 et gen=300 (population = 3) : f_best = -482

3. Algorithme génétique (geneic_mutation) qui est le plus efficace. C’est la solution que nous avons retenue : f_best = -430 ou meilleur que -430
