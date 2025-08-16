# Actor-Critic Based Sepsis Treatment Optimization

## Scenario

You are working as a data scientist in a medical AI research team. Your objective
is to train a reinforcement learning agent that can suggest optimal treatment strategies for
sepsis patients in the ICU. Sepsis is a life-threatening condition, and its management
requires timely interventions based on real-time vitals and clinical decisions. ICU patients
are monitored continuously, and treatment decisions (actions) are logged over time.

## Objective

The goal of this assignment is to model the ICU treatment process using Reinforcement
Learning, specifically the Actor-Critic method. The agent should learn an optimal
intervention policy from historical ICU data. Each patient's ICU stay is treated as an episode
consisting of time-stamped clinical observations and treatments.

Your tasks:
1. Model the ICU treatment process as a Reinforcement Learning (RL) environment.
2. Train an Actor-Critic agent to suggest medical interventions based on the patient’s
current state (vitals and demographics).

## Dataset

Use the dataset provided in the following link:
https://drive.google.com/file/d/1UPsOhUvyrsrC59ilXsvHwGZhzm7Yk01w/view?usp=sha
ring

## Features

● Vitals: mean_bp, spo2, resp_rate
● Demographics: age, gender
● Action: Medical intervention (e.g., "Vancomycin", "NaCl 0.9%", or NO_ACTION)
● Identifiers: timestamp, subject_id, hadm_id, icustay_id

## Environment Setup (RL Formulation)

### State Space

Each state vector consists of:
mean_bp (Mean Blood Pressure) , spo2 (Oxygen Saturation),
resp_rate (Respiratory Rate), age, One-hot encoded gender

### Action Space

● The agent selects one discrete action from 99 possible medical interventions (e.g.,
Vancomycin, Fentanyl, PO Intake, etc.
● You should integer encode or one-hot encode these interventions.

### Reward

At each time step, the agent receives a reward based on how close the patient's vitals are to
clinically normal ranges. The reward encourages the agent to take actions that stabilize the
patient's vital signs:
Rewardt= − ((MBPt−90)2+(SpO2t−98)2+(RRt−16)2)

### Explanation

● MBP (mean_bp): Target = 90 mmHg
● SpO₂ (spo2): Target = 98%
● RR (resp_rate): Target = 16 breaths/min
Each term penalizes the squared deviation from the healthy target. The smaller the
difference, the higher (less negative) the reward.

Example:
Suppose at time t, the vitals are:
● MBP = 88
● SpO₂ = 97
● RR = 20
Then the reward is:
Rewardt = − [(88−90)2+(97−98)2+(20−16)2] = − (4+1+16)= −21
A lower (more negative) reward indicates worse vitals, guiding the agent to learn actions
that minimize this penalty.

### Episode termination

An episode ends when the ICU stay ends. To define this:
1. Group the data by subject_id, hadm_id, icustay_id
→ Each group represents one ICU stay = one episode.
2. Sort each group by timestamp
→ Ensure the time progression is correct.
3. For each time step in a group (i.e., each row). Check if it is the last row in
that group.
→ If yes, then mark done = True (end of episode).
→ If no, then done = False (continue episode).
