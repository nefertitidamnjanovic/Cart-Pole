import gymnasium as gym
from enum import Enum
from random import Random
from dataclasses import dataclass, field
import numpy as np
from statistics import mean
from itertools import chain, repeat
import json
import os

random = Random()

CART_POS_RANGE = 2.4
CART_VEL_RANGE = 2.0
POLE_ANGLE_RANGE = .21
POLE_VEL_RANGE = 2.0

class Action(Enum):
    LEFT  = 0
    RIGHT = 1

    def __repr__(self) -> str:
        return 'LEFT' if self == Action.LEFT else 'RIGHT'

@dataclass
class State():
    cart_pos    : float = field(default_factory=float)
    cart_vel    : float = field(default_factory=float)
    pole_angle  : float = field(default_factory=float)

    def __post_init__(self):
        if self.cart_pos == -0.:
            self.cart_pos += 0.

        if self.pole_angle == -0.:
            self.pole_angle += 0.

        if self.cart_vel == -0.:
            self.cart_vel += 0.

    def __call__(self):
        return (self.cart_pos, self.cart_vel, self.pole_angle)

def update_policy(Q:dict[tuple, Action:int], policy:dict[tuple:Action]):
    ''' Finds the greedy policy with respect to the passed Q-values '''
    for state in Q.keys():
        policy[state] = max(Q[state], key=Q[state].get)

def init_Q(state:tuple, Q:dict, value:callable):
    '''Q-values are initialized with values from -1 to 0 for each action'''

    if state not in Q:
        Q[state] = {action: value() for action in Action}

def Q_learning(Q:dict[tuple,Action:int], state:State, action:Action, new_state:State, gain:int, gamma:float, alpha:float, exp_policy:dict[State:Action]) \
    -> Action:

    try:
        if new_state is not None:
            max_action = max(Q[new_state()], key=Q[new_state()].get)
            Q_aprox = gain + gamma*Q[new_state()][max_action]
            Q[state()][action] += alpha*(Q_aprox - Q[state()][action])

            return exp_policy[new_state()]

        Q[state()][action] += alpha*(gain - Q[state()][action])
    except Exception as e:
        init_Q(new_state(), Q, lambda: random.random()-1)
        init_Q(state(), Q, lambda: random.random()-1)
        update_policy(Q, exp_policy)

        return Q_learning(Q, state, action, new_state, gain, gamma, alpha, exp_policy)

def SARSA(Q:dict[State,Action:int], state:State, action:Action, new_state:State, gain:int, gamma:float, alpha:float, policy:dict[tuple:Action]):
    ''' Updates the policy with respect to the actual Q-values '''
 
    try:
        if new_state is not None:
            new_action = policy[new_state()]
            Q_aprox = gain + gamma*Q[new_state()][new_action]
            Q[state()][action] += alpha*(Q_aprox - Q[state()][action])
            return new_action 
        
        Q[state()][action] += alpha*(gain - Q[state()][action])
    except Exception as e:
        init_Q(new_state(), Q, lambda: random.random()-1)
        init_Q(state(), Q, lambda: random.random()-1)
        update_policy(Q, policy)

        return SARSA(Q, state, action, new_state, gain, gamma, alpha, policy)
            
def translate(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def all_states(cart_pos_prec:float, cart_vel_prec:float, pole_prec:float) -> list[State]:
    states = []
    for cart_pos in np.arange(-CART_POS_RANGE, CART_POS_RANGE+cart_pos_prec, 10**(-cart_pos_prec)):
        cart_pos = round(cart_pos, cart_pos_prec)

        for cart_vel in np.arange(-CART_VEL_RANGE, CART_VEL_RANGE+cart_vel_prec, 10**(-cart_vel_prec)):
            cart_vel = round(cart_vel, cart_vel_prec)

            for pole_angle in np.arange(-POLE_ANGLE_RANGE, POLE_ANGLE_RANGE+pole_prec, 10**(-pole_prec)):
                pole_angle = round(pole_angle, pole_prec)

                states.append(State(cart_pos, cart_vel, pole_angle))

    return states


def choose_method(methods:dict[str:tuple[int, float, float, callable]]) \
    -> tuple[int, float, float, callable]:
    ''' Returns parameters for the chosen method'''

    method_list = []
    for i, method in enumerate(methods.keys()):
        print(f"{str(i+1)}. {method}")
        method_list.append(method)

    prompts = chain(["Choose a method of learning for the agent: "], repeat(f"Method does not exist\nChoose another method: "))
    replies = map(input, prompts)
    method = next(filter(lambda s: s in methods.keys() or int(s) in range(1,len(methods.keys())+1), replies))
    if method.isnumeric():
        method = method_list[int(method)-1]

    return methods[method]

if __name__ == "__main__":

    methods = {                    
        "Q learning" : (5000, 0.9, 0.75, Q_learning),
        "SARSA"      : (5000, 0.9, 0.9, SARSA),
    }

    episode_count, alpha, gamma, learning_func = choose_method(methods)
    cart_pos_prec, cart_vel_prec, pole_angle_prec = (0, 1, 2)

    # Initializes the environment
    env = gym.make("CartPole-v1", render_mode="human")
    env.metadata["render_fps"] = 120000000000
    env.action_space.seed(82)
    observation, info = env.reset(seed=82)

    # Initializes the policy and Q-values with random actions and values
    states = all_states(cart_pos_prec, cart_vel_prec, pole_angle_prec)
    Q = {state() : {action : random.random()-1 for action in Action} for state in states}
    policy = {state() : random.choice([action for action in Action]) for state in states}

    avg_reward = [0] * 20

    for i in range(episode_count):
        reward_sum = 0

        observation, info = env.reset()
        cart_pos, cart_vel, pole_angle, _ = observation.astype(float) 
        state = State(round(cart_pos, cart_pos_prec), \
                    round(cart_vel, cart_vel_prec), \
                    round(pole_angle, pole_angle_prec))
        
        # First action of each episode is chosen randomly with 50% or from the current exploration policy
        action = random.choice([a for a in Action]) # policy[state()] if random.random() > 0.5 else random.choice([a for a in Action])

        # An episode is finished when the pole falls
        while 1: 
            observation, reward, terminated, truncated, info = env.step(action.value)
            if terminated or truncated:
                break

            cart_pos, cart_vel, pole_angle, _ = observation.astype(float) 
            new_state = State(round(cart_pos, cart_pos_prec), \
                            round(cart_vel, cart_vel_prec), \
                            round(pole_angle, pole_angle_prec))
            
            # The reward for each step is a function of the current pole angle and cart position
            reward_cart = translate(abs(cart_pos), 0, CART_POS_RANGE, 1, 0) 
            reward_pole = translate(abs(pole_angle), 0, POLE_ANGLE_RANGE, 1, 0) 
            reward = 0.65*reward_pole + 0.35*reward_cart

            action = learning_func(Q, state, action, new_state, reward, gamma, alpha, policy)

            # Each action has a chance of being random for obseration space exploration purposes
            if random.random() > 0.75:
                action = random.choice([a for a in Action])

            state = new_state

            reward_sum += round(reward)

        # After the end of an episode Q-values are updated for the last, non-terminal, state
        learning_func(Q, state, action, None, -1, gamma, alpha, policy)

        # The exploration policy is updated after each episode
        update_policy(Q, policy)

        if i > (episode_count-10):
            env.metadata["render_fps"] = 60
        elif i > (episode_count-50):
            env.metadata["render_fps"] = 120

        avg_reward.append(reward_sum)
        avg_reward.pop(0)
        print(f"Rewards: {mean(avg_reward):.2f}, Episode: {i}")
        
    env.close()

    out_policy = {str(state):repr(action) for state, action in policy.items()}
    out = {
        "episoded" : episode_count,
        "alpha"    : alpha,
        "gamma"    : gamma,
        "algorithm" : learning_func.__name__,
        "avg_reward" : mean(avg_reward),
        "policy" : out_policy,
    }

    filename = input("Enter filename: ")

    directory = os.getcwd() + "/Cart_pole/policies"
    if not os.path.exists(directory):
        os.makedirs(directory)

    json_obj = json.dumps(out, indent=2)
    with open(directory + "/" + filename + ".json", "w+") as outfile:
        outfile.write(json_obj)

    




