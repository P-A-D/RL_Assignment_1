# RL_Assignment_1
The repository for assignment #1 of the Reinforcement Learning course - Spring 2024

Student names: Pedram A. Darestani - Zahra Jabari

Student ids:   202383919           - 202291677

In order to replicate the results, you can clone this repository and use the templates given in the files Runs1.ipynb, Runs2.ipynb, and Runs2_cross_model_eval.ipynb.

These three notebooks call the two Part1.py and Part2.py scripts that train and evaluate agents.

All notebooks have markdown chunks that explain what the code chunk below them does.

You can adjust the arguments passed to the agent/evaluator objects in order to further adjust their behavior.

The code usage in the notebooks is quite short and straightforward.

Most of the possible customizations are implemented in the notebooks and by following the code chunks and markdown above them, you can easily understand how they work.

Both Runs2.ipynb and Runs2_cross_model_eval.ipynb notebooks are for the seconds part of the assignment. Their difference is in the initial reward mean distribution.

In the Runs2.ipynb, every model was trained 1000 times for 20,000 time steps, while all of these 1000 runs used the same reward distribution.
This allows the performance of the 1000 models to be aggregated. However, when another model is being trained 1000 times for 20,000 time steps, while all of them
share the same initial reward distribution, this reward distribution is different from the ones for the previously trained model. As a result, one could argue
that the performance of two models, both of which have been trained 1000 times for 20,000 time steps each, is not comparable since they had different initial
reward distributions. 

In the Runs2_cross_model_eval.ipynb notebook an attempt was made to reduce the impact of this matter, and all different models were trained on one shared
random initial reward distribution. Although this can reduce the bias in comparison, it would be better if this was done 1000 different times and then the
results were aggregated in order to consider the effects of different reward mean initializations on the performance of models in non-stationary reward problems.
