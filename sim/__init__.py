"""ManiSkill3 simulation module for RM evaluation and RM-guided policy training.

Two evaluation axes for the Reward Model:
    1. **RM-guided SAC training** (``sac_train``) — train a vision-based policy
       with RM reward shaping, measure task success rate.
    2. **Trajectory evaluation** (``traj_eval``) — score real rollout trajectories
       with the RM, measure PRA / monotonicity / correlation.
"""
