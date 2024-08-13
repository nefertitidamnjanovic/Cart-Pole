from cart_pole import State, Action
import gymnasium as gym
import json
from itertools import chain, repeat
import os

def continue_question(msg : str) -> bool:
    prompts = chain([msg], \
                    repeat(f"Only 'y' or 'n' are valid responses. \n{msg}"))
    replies = map(input, prompts)

    valid = next(filter(lambda s: s == 'y' or s == 'n', replies))

    print("------------------------------")

    if valid == 'y':
        return True
    return False

if __name__ == "__main__":

    env = gym.make("CartPole-v1", render_mode="human")
    env.metadata["render_fps"] = 60
    env.action_space.seed(82)
    observation, info = env.reset(seed=82)

    try:
        directory = os.getcwd() + "/Cart_pole/policies/"
        file = input("Name of file with policy: ")
        file += ".json"

        with open(directory + file) as in_file:
            policy_dict = json.load(in_file)
            policy = {eval(state): Action.LEFT if action == "LEFT" else Action.RIGHT for state, action in policy_dict["policy"].items()}
            print(f"Average reward: {policy_dict['avg_reward']}")

            cart_pos_prec, cart_vel_prec, pole_angle_prec = (0, 1, 2)

            while continue_question("Run? "):
                reward_sum = 0

                observation, info = env.reset()
                cart_pos, cart_vel, pole_angle, _ = observation.astype(float) 
                state = State(round(cart_pos, cart_pos_prec), \
                            round(cart_vel, cart_vel_prec), \
                            round(pole_angle, pole_angle_prec))
                
                action = policy[state()]

                while 1: 
                    observation, reward, terminated, truncated, info = env.step(action.value)
                    if terminated or truncated:
                        break

                    cart_pos, cart_vel, pole_angle, _ = observation.astype(float) 
                    state = State(round(cart_pos, cart_pos_prec), \
                                    round(cart_vel, cart_vel_prec), \
                                    round(pole_angle, pole_angle_prec))
                    
                    action = policy[state()]

                    reward_sum += round(reward)

                print(f"Reward: {reward_sum}")

    except Exception as e:
        print(e)
