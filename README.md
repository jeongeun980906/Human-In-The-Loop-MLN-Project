# Human-In-The-Loop-Project with MLN

This task is non-prehensile manipullation task,
Inspired by "Non-Prehensile Manipulation in Clutter with Human-In-The-Loop" (ICRA 2020)\\
The details about this paper is in [Paper](https://pubs.rpapallas.com/icra2020/).

I divided global task planner with "Mixture of Logit Networks" and heuristic local planner.
Mixture of Logit Networks (i.e., MLN) can acheive both epistemic uncertanty and aleatoric uncertainty which makes 
humman annotators to stop the robot's movement and give robot a direct command.

For more details, the presentation url is here.
[Presentation Video](https://www.youtube.com/watch?v=uXTpxWBCBlA&t=276s)

## Results
**Toy Task**
![Toy task](https://github.com/jeongeun980906/Human-In-The-Loop-MLN-Project/blob/master/videos/toy_task.gif)

**Main Task**
![Main task](https://github.com/jeongeun980906/Human-In-The-Loop-MLN-Project/blob/master/videos/main_task.gif)

## Requirements
- pybullet
- pytorch
- attrdict
- matplotlib
- opencv-python

## Train humman in the loop
<code>
python main.py
</code>

## Test Heuristic Planner
<code>
python heurstic.py
</code>
