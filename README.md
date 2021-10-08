# CAPS

This project implements CAPS: Comprehensible Abstract Policy Summaries

To run CAPS for an implemented environment:

pip install -r requirements.txt

python run.py --env='env_name' --path='model_path' --other_flags=...

Ex: python run.py --env=mountain --path=./MountainCar/checkpoints/checkpoint-580 --num_episodes=3 --alg=PPO

To run CAPS with a new environment:
1. Train the environment separately. Save the tensorflow or pytorch model.
2. Implement a function which will run the agent in testing and return the dataset, D, specified in the paper.
3. Alter run.py to include your test function, and handle env and path flags appropriately.
4. Create a class in translation.py that includes your user-predicates, following the template laid out in the code.
5. Run CAPS
