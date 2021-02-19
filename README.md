[//]: # (Image References)

[image1]: single_agent_solved.png "Trained Agent"

# Project 2: Continuous Control

### Introduction

This project trains a single agent to work with the Reacher environment (first version). The Reacher environment simulates the controlling of a double-jointed arm, to reach target locations.

#### The Environment
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

![Trained Agent][image1]

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes

### Getting Started
#### Step 1: Clone the DRLND Repository

Follow the [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. These instructions can be found in README.md at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

(For Windows users) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

#### Step 2: Download the Unity Environment

For this project, you will **not** need to install Unity - this is because the environment is already built for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

Version 1: One (1) Agent
 - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
 - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
 - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
 - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip) - **Note: The Agent was implemented and trained on this version !**

###### Currently Version 2 is not implemented (right now will not work, but is available for future implementations):
Version 2: Twenty (20) Agents
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Then, place the file in the `p2_continuous-control/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.

(For Windows users) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

### Instructions

Follow the instructions in `Continuous_Control_PPO.ipynb` to train your own agent or test the already trained agent:
1. Start the Environment - currently works only with version 1 (single agent) of the environment
2. Examine the State and Action Spaces
3. Optionally take Random Actions in the Environment
4. Train the Agent (Check Further Modifications section or optionally you can skip to 5. Testing)
5. Test trained Agent

### Further Modifications

In searching for better performance you can modify:
1. Training process by modifying `hyperparameters` (check Report.md for my hyperparameters search history)

        'episode_count': 1500,
        'discount_rate': 0.99,
        'tau': 0.5,
        'gradient_clip': 15,
        'buffer_size': 3072,
        'optimization_epochs': 2,
        'ppo_clip': 0.2,
        'batch_size': 64,
        'adam_learning_rate': 3e-4,
        'adam_epsilon': 1e-4

 2. Modifying any part of Agent's Neural Network's Policy architecture in section 4.1.

        Actor = [state_size, 256] -> ReLU -> [ 64] -> ReLU -> [action_size] -> tanh
        Critic =[state_size, 512] -> ReLU -> [256] -> ReLU -> [64] -> [1]
        PPO_Policy - Normal Distribution with Standard Deviation of 1 

3. Implementing different Policy Search Algorithm (like DDPG, A2C and others)
4. Implement Multiagent version.
5. It would be interesting to try [Attention](https://arxiv.org/abs/1706.03762) mechanism for this kind of problem.

For more information check the [Report.md](Report.md)