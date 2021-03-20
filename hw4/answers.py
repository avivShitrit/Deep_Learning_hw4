r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(
        batch_size=16, gamma=0.95, beta=0.2, learn_rate=1e-3, eps=1e-8, num_workers=2,
    )
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======
    
    # ========================
    return hp


def part1_aac_hyperparams():
    hp = dict(
        batch_size=8,
        gamma=0.95,
        beta=1.,
        delta=1.,
        learn_rate=1e-3,
        eps=1e-8,
        num_workers=2,
    )
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    # ====== YOUR CODE: ======
    
    # ========================
    return hp


part1_q1 = r"""
**Your answer:**
When we subtract the baseline from the reward we get the advantage which is the measure of how much the current action is better than
the usual action we would take in the same state. The reason this subtraction help reduce the variance is that the advantage has lower
variance from because the baseline components for the variance introduced since its not depand on the state.
Example: let assume we have very complicated task where for similar states we can very different rewards, by subtracting the baseline
we are reducing the variance which will then by using experience will converge to a good behavior
"""


part1_q2 = r"""
**Your answer:**
The reason we always get a valid approximation is that the connection between the terms $v_\pi(s)$ and $q_\pi(s,a)$. 
Actually, $v_\pi(s)$ can be expressed as $\sum_{a \in A}\pi(a|s)q_\pi(s,a)$ which means that $v_\pi(s)$ is actually 
takes into consideration all the possible q-values given state $s$ with respect to the agent behavior. 

So if we define the baseline that way it will give us good results, since q-values close to this baseline will lead to lower 
loss values just like we want.

"""


part1_q3 = r"""
**Your answer:**
1. If we examine the mea_reward graph we can see that there is no major difference between the experiments they all got 
to approximately 50 mean_reward after 4000 episode. The best model was the cpg as we can see he got the best mean_reward 
above the others and he also climbed early and picked at 2500 episodes
We can also see that most fast learner was the bpg model that picked at 2000 episodes before all the rests.

2. The results we got was very convincing that the aac model was far better then the rest. his best mean_reward was about 100
and it looks like it can still climb if we train it longer where all the other models remain steady in approximately 50 mean_reward
"""
