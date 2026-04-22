# G1 Pick-and-Place: A Hybrid Hierarchical Approach

## Result

The results are relatively good, the highly tuned teacher and RL training method allowed us to bypass the multi-hour required training time while the reward tunings taught the robot how to run the routine. It reached a 100% accuracy with the exact setup given in the run.py. Though we used a teacher in the model provided in this repo it can be fully trained without the teacher through the existing reward based RL system that uses Proximal Policy Optimization. 

Evaluated via `make run-headless RUNS=50` on the train-teacher-high spawn window (8 demos, 600 epochs, around 5 min CPU training), 50/50 successes where success is the `placed` predicate in sim.py:382-389.

In our implementation here, we use a simple method of staving off ending up with a new policy which is too far from the old policy. It is just early stopping. If the mean KL-divergence of the new policy from the old grows beyond a threshold, we stop taking gradient steps. (train.py:596,645,653-655)

sim.py - simulation enviorment based on run.py
train.py - base PPO algorithm (task-agnostic)
demo.py - task definition: reward/curriculum, scripted teacher, training CLI

## Approach

I treated this as a coordination problem rather than a full-body control problem. The provided walker and right-arm reacher already solve the hard low-level control; what's missing is a layer that decides when to walk, where to aim the hand, and when to grip. So I built that layer on top rather than replacing the priors.

The environment exposes a 7D task-level action space: forward/lateral velocity, yaw rate, three hand-target deltas, and a grip command.

The policy issues commands in that space while the pretrained skills handle joint-level execution. A scripted finite-state teacher drives the seven phases, using the head camera for alignment and the wrist camera near the grasp where local alignment dominates. Training is PPO (Proximal Policy Optimization, https://spinningup.openai.com/en/latest/algorithms/ppo.html) with the provided camera views.


## What I Learned

Essentially the core challenge I ran into was evaluvating the best solution to attack this problem. I considered two alternatives before landing on the hybrid architecture. End-to-end joint-level RL from scratch would balloon the search space making training be a multi hour nightmare as it is difficult to debug a silent enviorment. A purely scripted controller would overfit to one placement and make no use of the cameras. The hybrid is able to train with existing knowledge of the routine, use the provided vision cameras, and still leaves room for the policy to improve past the teacher. 

## Answering Questions

**How do you decompose this into phases (approach, grasp, transport, place)? Can they overlap?**

Seven phases. Approach, pregrasp, grasp, lift, transport, place, release. Phase is a single string on the scripted teacher (demo.py:223) with transitions driven by thresholded metrics (demo.py:267-293). Phases do not overlap, each control step is in exactly one phase. The learned policy is not phase-aware at inference, it receives observations and emits a 7D action, so blending across teacher phase boundaries is available to it.

**How do you coordinate locomotion and manipulation? Does the robot need to stop walking to reach?**

Both share one action vector. Dims 0-2 drive walker base linear velocity and yaw rate (sim.py:511-513). Dims 3-5 offset the pelvis-frame reach target for the reacher. Dim 6 is grip command (sim.py:526). Walker and reacher run every step (sim.py:109,112). Concurrent walk-and-reach is supported by the stack. The scripted teacher stops base motion from pregrasp onward (demo.py:323-336) because stationary grasping is empirically more reliable, not because the stack requires it.

**The provided reacher targets a point in pelvis frame — how do you decide what to target and when?**

Target is block position relative to the base plus a phase-specific offset. Pregrasp uses a standoff above the block (demo.py:331-333). Grasp uses an offset at block height (demo.py:339). Transport uses a fixed carry pose (demo.py:366). Place uses the target-table position plus a hover offset, collapsing to a drop offset when close (demo.py:372-374). The reach command is a proportional step from the current target toward the desired pose (demo.py:233-235). For the learned policy the target is implicit, the policy outputs reach deltas directly.

**How would you use the wrist camera? The head camera? Do you need them?**

Both are used with the head camera detecting the red cylinder at approach distance for coarse heading alignment (demo.py:260, 315-317). Wrist camera is used at pregrasp and grasp for local lateral centering where head framing becomes unreliable (demo.py:261, 328-330, 340-342). Detection is a red-color threshold (demo.py:237-249) while the policy network ingests both images. Is it strictly needed though...kinda but not really as the proprio vector already contains block-relative position (sim.py:449-461).

**How do you verify the cylinder is actually grasped before trying to move How do you verify it's placed?**

Grasp is verified by grasp_score (sim.py:373-380), a sum of finger-contact count capped at 1.0 plus bonuses for closed grip and sustained contact, clipped to [0, 1]. The teacher requires grasp_score above threshold and palm_to_block under 0.07 before transitioning to lift (demo.py:280). Placement is verified by the placed predicate (sim.py:382-389): target surface supporting the block, robot not in contact with the block, block horizontal position inside the target footprint with a 1.5cm margin, block bottom within 3cm of the target surface, and block speed below 0.2 m/s. The speed check prevents counting a mid-bounce block as placed.

**Where does sim-to-real transfer factor into your thinking (even if you're not doing it)?**

There are three gaps currently: the red-color detector (demo.py:237-249) is brittle to lighting and would need a learned segmenter before real deployment. Grasp score uses simulator contact bodies (sim.py:373-380), reality would substitute tactile or force feedback. Spawn distribution for the demo is narrowed to ensure it can be reproduced which is the opposite of what transfer requires. A real deployment would need domain randomization over pose, lighting, and object appearance. Also for a real sim-to-real transfer I would need to on a GPU cluster which is a resource I unfortunately currently don't have on hand (though I would love to have a GPU cluster). 

## Limitations and Next Steps

The biggest weaknesses are all in the hand-designed scaffolding as the object detector is red-color thresholding, the grasp score is a sim-side heuristic, and the teacher is a fixed FSM. Reasonable for a time limited solution, but bad as it ties this exact scene. Though the framework itself in train.py and sim.py are agnostic to the particular task we would need to create a proper reward mechanism for whatever routine we would want to train.

To improve this particular demo if given more time:
1. Replace color thresholding with a learned detector or segmentation model as this is the most brittle piece and the first thing that would break under visual variation.
2. Learn a grasp-quality estimator so the policy can distinguish contact from a secure grasp.
3. Add pose and visual randomization during training to reduce scene overfitting.