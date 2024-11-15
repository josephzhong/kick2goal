'''
Version of 2v0 attacker training from Adam's research.

Command to replicate results: 
Note you can change num_cpus and vec_envs if you don't have a lot of cores. It simply speeds up training.

python run.py --train --env=skillTrain --train_steps=30000000 --num_cpus=16 --vec_envs=32 --batch_size=16384 --wandb
'''

import copy
import functools
import time
from pettingzoo import ParallelEnv
import gymnasium as gym
import pygame
import math
import numpy as np
import random
import sys

sys.path.append(sys.path[0] + "/..")


def env(render_mode=None, curriculum_method="close"):
    env = parallel_env(curriculum_method=curriculum_method)
    return env


class parallel_env(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode="rgb_array", curriculum_method="close"):
        self.rendering_init = False
        self.render_mode = render_mode
        self.curriculum_method = curriculum_method

        ## Evaluation settings ##

        # Domain randomization
        self.action_noise = 0.0
        self.observation_noise = 0
        self.ball_contact_angle_noise = np.pi/1.5
        self.ball_contact_distance_noise = 2.5

        ####
        
        self.terminate_dict = {
            "goal": False,
            "out_of_bounds": False,
        }

        self.reward_dict = {
            "goal": 3000,  # Team
            "out_of_bounds": 0,  # Team
            "ball_to_goal": 0.5,  # Team
            "agent_to_ball": 0.3,  # Team
            "kick": 6,  # Individual
            "missed_kick": -5,  # Individual
            "movement_penalty": -0.4,
            "position_penalty": -1,
            "facing_ball": 0.5,
            "position_reward": 0.3,
            "robot_out_of_bounds": 0,
            "too_close": -50,
            "time_step": 0,
        }

        # Parameters
        self.episode_length = 1000

        self.goal_size = 500

        self.ball_acceleration = -0.8
        self.ball_velocity_coef = 4
        self.last_touched = None

        self.displacement_coef = 0.15
        self.angle_displacement = 0.2

        self.robot_x_size = 300
        self.robot_y_size = 400
        self.ball_radius = 10

        self.num_defenders = 0
        self.defenders = []
        self.defender_info = []

        # agents
        self.num_robots = 2

        self.possible_agents = list(range(self.num_robots))
        self.agents = self.possible_agents[:]

        # 3D vector of (x, y, angle, kick) velocity changes
        action_space = gym.spaces.Box(
            np.array([-1, -1, -1, -1, -1]),
            np.array([1, 1, 1, 1, 1]),
        )
        self.action_spaces = {agent: action_space for agent in self.agents}

        # observation spaces
        history_length = 3
        self.obs_history = {agent: History(history_length) for agent in self.agents}

        self.reset()
        obs_size = len(self.get_obs(0, include_history=True))

        observation_space = gym.spaces.Box(low=-1, high=1, shape=(obs_size,))
        self.observation_spaces = {agent: observation_space for agent in self.agents}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def get_distance(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))

    """
    Takes as input robot or opponent robot NAME
    """
    
    def can_kick(self, robot, distance=200):
        robot_loc = self.robots[robot]

        if self.check_facing_ball(robot):
            return (
                self.get_distance(robot_loc, self.ball)
                < distance
            )

    """
    ego-centric observation:
        origin,
        goal,
        other robots,
        ball
    """

    def get_obs(self, robot, include_history=False):
        robot_loc = self.robots[robot]
        obs = []

        # Ball
        ball = self.get_relative_observation(robot_loc, self.ball)
        obs.extend(ball)

        # 1 hot for can kick
        obs.extend([1] if self.can_kick(robot, distance=150) else [0])

        # Teammates
        for teammate in self.agents:
            if robot != teammate:
                teammate_loc = self.robots[teammate]
                # Add noise to teammate location
                teammate_loc = [
                    teammate_loc[0],
                    teammate_loc[1],
                ]
                obs.extend(self.get_relative_observation(robot_loc, teammate_loc))

        # Goal
        # goal = self.get_relative_observation(robot_loc, [4500, 0])
        # obs.extend(goal)

        goalPost1 = self.get_relative_observation(robot_loc, [4500, self.goal_size])
        obs.extend(goalPost1)

        goalPost2 = self.get_relative_observation(robot_loc, [4500, -self.goal_size])
        obs.extend(goalPost2)

        opponentGoalPost1 = self.get_relative_observation(robot_loc, [-4500, self.goal_size])
        obs.extend(opponentGoalPost1)

        opponentGoalPost2 = self.get_relative_observation(robot_loc, [-4500, -self.goal_size])
        obs.extend(opponentGoalPost2)

        slide1 = self.get_relative_observation(robot_loc, [0, 3000])
        obs.extend(slide1)

        slide2 = self.get_relative_observation(robot_loc, [0, -3000])
        obs.extend(slide2)

        # Defenders
        for defender in self.defenders:
            obs.extend(self.get_relative_observation(robot_loc, defender))

        if include_history:
            for prev_obs in self.obs_history[robot].get():
                obs.extend(prev_obs)

        return obs

    """
    Format for robots:
    x, y, angle

    Format for robot velocities:
    x, y, angle

    Format for ball:
    x, y, velocity, angle
    """

    def reset(self, seed=None, return_info=False, options=None, **kwargs):
        self.time = 0
        
        far_robot_x = np.random.uniform(-4500, -0)
        far_robot_y = np.random.uniform(-3000, 3000)
        far_robot_angle = np.random.uniform(-np.pi, np.pi)

        close_robot_x = np.random.uniform(0, 4500)
        close_robot_y = np.random.uniform(-3000, 3000)
        close_robot_angle = np.random.uniform(-np.pi, np.pi)

        # far_robot_x = -2000
        # far_robot_y = -1000
        # far_robot_angle = np.pi/2

        # close_robot_x = 3000
        # close_robot_y = 1
        # close_robot_angle = -np.pi/2

        # Robots
        self.robots = [
            [far_robot_x, far_robot_y, far_robot_angle],
            [close_robot_x, close_robot_y, close_robot_angle]
        ]

        if np.random.uniform() < 0.5:
            ball_x = np.clip(close_robot_x + np.random.uniform(-1000, 1000), -4400, 4400)
            ball_y = np.clip(close_robot_y + np.random.uniform(-1000, 1000), -2900, 2900)
        else:
            ball_x = np.clip(far_robot_x + np.random.uniform(-1000, 1000), -4400, 4400)
            ball_y = np.clip(far_robot_y + np.random.uniform(-1000, 1000), -2900, 2900)

        # ball_x = np.random.uniform(-4500, 4500)
        # ball_y = np.random.uniform(-3000, 3000)
        # ball_x = -3000
        # ball_y = 0

        self.ball = [ball_x, ball_y, 0, 0]

        
        # # Override for testing
        # self.robots = [[-3000, -1000, 90 * np.pi / 180], 
        #                [1500, 1000, -90 * np.pi / 180]]
        
        # self.ball = [-3000, 0, 0, 0]
            

        self.terminate_dict["goal"] = False
        self.terminate_dict["out_of_bounds"] = False

        for agent in self.agents:
            for _ in range(self.obs_history[agent].getMaxLength()):
                history = []
                history.extend(self.get_relative_observation(self.robots[agent], self.ball))
                self.obs_history[agent].add(history)

        observations = {}
        for agent in self.agents:
            observations[agent] = self.get_obs(agent, include_history=True)

        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, actions):
        """w
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        obs, rew, terminated, truncated, info = {}, {}, {}, {}, {}
        self.time += 1

        # Copy previous locations (deep copy)
        self.prev_ball = copy.deepcopy(self.ball)
        self.prev_agents = copy.deepcopy(self.robots)

        # Update agent locations and ball
        for agent in self.agents:
            action = actions[agent]
            self.move_agent(agent, action)

        self.update_ball()

        # Calculate rewards
        for agent in self.agents:
            obs[agent] = self.get_obs(agent, include_history=True)
            rew[agent] = self.calculate_reward(agent, actions[agent])
            terminated[agent] = self.time > self.episode_length
            truncated[agent] = False
            info[agent] = {}

        if self.terminate_dict["goal"] or self.terminate_dict["out_of_bounds"]:
            terminated = {agent: True for agent in self.agents}

        for agent in self.agents:
            # Create history
            history = []
            # add ball position
            history.extend(self.get_relative_observation(self.robots[agent], self.ball))

            self.obs_history[agent].add(history)

        # self.terminate_dict["out_of_bounds"] = False

        return obs, rew, terminated, truncated, info

    """
    Checks if ball is in goal area
    """

    def goal(self):
        # if self.ball[0] > 4500 and self.ball[1] < 750 and self.ball[1] > -750:
        if self.ball[0] > 4500 and self.ball[1] < self.goal_size and self.ball[1] > -self.goal_size:
            return True
        return False

    """
    Get angle for contacting other robot
    """

    def get_angle(self, pos1, pos2):
        return np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0])

    def calculate_reward(self, agent, action):
        reward = 0
        
        # Facing ball
        if self.check_facing_ball(agent, req_angle=40):
            reward += self.reward_dict["facing_ball"]

        # Order robots from cloest to goal to furthest
        robot_distances = []
        for robot in self.agents:
            robot_distances.append(
                self.get_distance(self.robots[robot], [4800, 0])
            )
        robot_order = np.argsort(robot_distances)
        
        # Attempting to kick
        if action[3] > 0.6:
            if self.can_kick(agent):
                # Successful kick
                reward += self.reward_dict["kick"]
            else:
                # Missed kick
                reward += self.reward_dict["missed_kick"]
            
        # Goal
        if self.goal():
            reward += self.reward_dict["goal"]
            self.terminate_dict["goal"] = True
            
            
        if robot_order[0] == agent:
            target = [4500, 0]
        else:
            # Target should be 200 units closer to goal than closest robot
            # targetx = self.robots[robot_order[0]][0] + 400 * np.cos(self.get_angle(self.robots[robot_order[0]], [4500, 0]))
            # targety = self.robots[robot_order[0]][1] + 400 * np.sin(self.get_angle(self.robots[robot_order[0]], [4500, 0]))

            # target = [targetx, targety]
            targetx = self.robots[robot_order[0]][0] + 200
            targety = 0
            target = [targetx, targety]


        # Ball to target
        reward += self.reward_dict["ball_to_goal"] * (
            self.get_distance(self.prev_ball, target)
            - self.get_distance(self.ball, target)
        )

        # Order robots from closest to ball to furthest
        agent_ball_distances = []
        for robot in self.agents:
            agent_ball_distances.append(
                self.get_distance(self.robots[robot], self.ball)
            )
        agent_ball_order = np.argsort(agent_ball_distances)

        # Get distance between agent and closest to ball
        distance_robot_closest = self.get_distance(self.robots[agent], self.robots[agent_ball_order[0]])
        if distance_robot_closest < 300 and agent != agent_ball_order[0]:
            reward += self.reward_dict["too_close"]

        # if closest to ball
        if agent == agent_ball_order[0]:
            reward += self.reward_dict["agent_to_ball"] * (
                self.get_distance(self.prev_agents[agent], self.ball) - 
                self.get_distance(self.robots[agent], self.ball)
            )
        
        if self.robot_out_of_bounds(agent):
            reward += self.reward_dict["robot_out_of_bounds"]

        # if self.terminate_dict["out_of_bounds"]:
        #     reward += self.reward_dict["out_of_bounds"]

        # Punish for movement
        if action[4] < 0 and agent != agent_ball_order[0]:
            reward += self.reward_dict["movement_penalty"]

        # if agent != agent_ball_order[0] and np.abs(self.robots[agent][1]) > 800:
        #     reward += self.reward_dict["position_penalty"]

            # reward += self.reward_dict["position_reward"] * (
            #     np.abs(self.robots[agent][1] - self.prev_agents[agent][1])
            # )

        reward += self.reward_dict["time_step"]

        return reward
    
    def robot_out_of_bounds(self, robot):
        if (
            self.robots[robot][0] > 4600
            or self.robots[robot][0] < -4600
            or self.robots[robot][1] > 3100
            or self.robots[robot][1] < -3100
        ):
            return True
        return False

    #### STEP UTILS ####
    def scale_action(self, action):
        for v in range(len(action)):
            if abs(action[v]) < 0.0:
                action[v] = 0
            else:
                action[v] = np.sign(action[v]) * (abs(action[v]) - 0.0)
        
        clips = [
            [1, -0.3],
            [0.5, -0.5],
            [1 -1],
                 ]
        # clip
        for i in range(len(clips[0])):
            action[i] = np.clip(action[i], clips[i][1], clips[i][0])

        # Add noise to action (not kick)
        for i in range(len(action) -  1):
            action[i] += np.random.normal(0, self.action_noise)
        
        return action
    

    def move_agent(self, robot, action):
        if action[4] > 0:
            # Stand still
            return
        if action[3] > 0.6:
            # Kick the ball
            self.kick_ball(robot, action[3])
        else:
            # Clip velocities
            robot_move_values = self.scale_action(
                action
            )
            
            policy_goal_x = self.robots[robot][0] + (
                (
                    (np.cos(self.robots[robot][2]) * robot_move_values[0])
                    + (
                        np.cos(self.robots[robot][2] + np.pi / 2)
                        * robot_move_values[1]
                    )
                )
                * 100
            )  # the x component of the location targeted by the high level action
            policy_goal_y = self.robots[robot][1] + (
                (
                    (np.sin(self.robots[robot][2]) * robot_move_values[0])
                    + (
                        np.sin(self.robots[robot][2] + np.pi / 2)
                        * robot_move_values[1]
                    )
                )
                * 100
            )  # the y component of the location targeted by the high level action

            # Update robot position
            self.robots[robot][0] = (
                self.robots[robot][0] * (1 - self.displacement_coef)
                + policy_goal_x * self.displacement_coef
            )  # weighted sums based on displacement coefficient
            self.robots[robot][1] = (
                self.robots[robot][1] * (1 - self.displacement_coef)
                + policy_goal_y * self.displacement_coef
            )  # the idea is we move towards the target position and angle

            # Update robot angle
            self.robots[robot][2] = (
                self.robots[robot][2]
                + robot_move_values[2] * self.angle_displacement
            )

            # # Check for collisions with other robots from current action
            # for other in self.agents:
            #     if other == robot:
            #         continue
            #     self.check_collision(robot, other)

            # for other in range(self.num_defenders):
            #     self.check_collision(robot, other, defender=True)

        # Make sure robot is on field
        self.robots[robot][0] = np.clip(self.robots[robot][0], -5200, 5200)
        self.robots[robot][1] = np.clip(self.robots[robot][1], -3700, 3700)
        

    def check_collision_ball(self, robot):
        # Unpack robot and ball properties
        robot_x, robot_y, robot_angle = self.robots[robot]
        ball_x, ball_y, _, _ = self.ball

        # Rotate ball's center point back to axis-aligned
        dx = ball_x - robot_x
        dy = ball_y - robot_y
        rotated_x = robot_x + dx * math.cos(-robot_angle) - dy * math.sin(-robot_angle)
        rotated_y = robot_y + dx * math.sin(-robot_angle) + dy * math.cos(-robot_angle)

        # Closest point in the rectangle to the center of circle rotated backwards (unrotated)
        closest_x = min(robot_x + self.robot_x_size / 2, max(robot_x - self.robot_x_size / 2, rotated_x))
        closest_y = min(robot_y + self.robot_y_size / 2, max(robot_y - self.robot_y_size / 2, rotated_y))

        # Re-rotate the closest point back to the rotated coordinates
        dx = closest_x - robot_x
        dy = closest_y - robot_y
        closest_x = robot_x + dx * math.cos(robot_angle) - dy * math.sin(robot_angle)
        closest_y = robot_y + dx * math.sin(robot_angle) + dy * math.cos(robot_angle)

        # Calculate the distance between the circle's center and this closest point
        distance = math.sqrt((closest_x - ball_x) ** 2 + (closest_y - ball_y) ** 2)
        
        # If the distance is less than the ball's radius, an intersection occurs
        collision = distance <= self.ball_radius
        
        if collision:
            # Normalize the direction vector by the robot's width and height
            direction_dx = (closest_x - robot_x) / self.robot_x_size
            direction_dy = (closest_y - robot_y) / self.robot_y_size

            # Calculate the angle between the normalized direction vector and the robot's orientation
            direction_angle = math.atan2(direction_dy, direction_dx) - robot_angle

            # Normalize the angle to the range [-pi, pi]
            direction_angle = (direction_angle + math.pi) % (2 * math.pi) - math.pi

            # Determine the side of the collision based on the direction angle
            if -math.pi / 4 <= direction_angle < math.pi / 4:
                direction = [1, 0]
            elif math.pi / 4 <= direction_angle < 3 * math.pi / 4:
                direction = [0, 1]
            elif -3 * math.pi / 4 <= direction_angle < -math.pi / 4:
                direction = [0, -1]
            else:
                direction = [-1, 0]

            # Rotate the direction vector back to global coordinates
            direction_x = direction[0] * math.cos(robot_angle) - direction[1] * math.sin(robot_angle)
            direction_y = direction[0] * math.sin(robot_angle) + direction[1] * math.cos(robot_angle)

            # Convert the direction vector to an angle in radians
            angle = math.atan2(direction_y, direction_x)
        else:
            angle = None

        return collision, angle
    
    def check_collision_robots(self, robot1, robot2):
        #  Unpack robot properties
        robot1_x, robot1_y, robot1_angle = self.robots[robot1]
        robot2_x, robot2_y, robot2_angle = self.robots[robot2]

        # Rotate robot2's center point back to axis-aligned with robot1
        dx = robot2_x - robot1_x
        dy = robot2_y - robot1_y
        rotated_x = robot1_x + dx * math.cos(-robot1_angle) - dy * math.sin(-robot1_angle)
        rotated_y = robot1_y + dx * math.sin(-robot1_angle) + dy * math.cos(-robot1_angle)

        # Closest point in the rectangle to the center of robot2 rotated backwards (unrotated)
        closest_x = min(robot1_x + self.robot_x_size / 2, max(robot1_x - self.robot_x_size / 2, rotated_x))
        closest_y = min(robot1_y + self.robot_y_size / 2, max(robot1_y - self.robot_y_size / 2, rotated_y))

        # Re-rotate the closest point back to the rotated coordinates
        dx = closest_x - robot1_x
        dy = closest_y - robot1_y
        closest_x = robot1_x + dx * math.cos(robot1_angle) - dy * math.sin(robot1_angle)
        closest_y = robot1_y + dx * math.sin(robot1_angle) + dy * math.cos(robot1_angle)

        # Calculate the distance between the robot2's center and this closest point
        distance = math.sqrt((closest_x - robot2_x) ** 2 + (closest_y - robot2_y) ** 2)

        # If the distance is less than the sum of half of the robots' sizes, an intersection occurs
        return distance <= (self.robot_x_size + self.robot_y_size) / 2

    def update_ball(self):
        # Update ball velocity
        self.ball[2] += self.ball_acceleration
        self.ball[2] = np.clip(self.ball[2], 0, 100)

        # Update ball position
        self.ball[0] += self.ball[2] * math.cos(self.ball[3])
        self.ball[1] += self.ball[2] * math.sin(self.ball[3])

        # If ball touches robot, push ball away
        for agent in self.agents:
            collision, angle = self.check_collision_ball(agent)
            if collision:
                self.ball[2] = self.ball_velocity_coef * 10
                self.ball[3] = angle + np.random.uniform(-self.ball_contact_angle_noise, self.ball_contact_angle_noise)

        # If ball OOB, terminate
        if abs(self.ball[0]) > 4500 or abs(self.ball[1]) > 3000:
            # If not in goal area
            if not (self.ball[0] > 4500 and self.ball[1] < self.goal_size and self.ball[1] > -self.goal_size):

                # # Move ball to closest point in bounds and set velocity to 0
                # self.ball[0] = np.clip(self.ball[0]-1000, -4400, 4400)
                # self.ball[1] = np.sign(self.ball[1]) * 2900
                # self.ball[2] = 0

                self.terminate_dict["out_of_bounds"] = True


    def check_facing_ball(self, robot, req_angle=10):
        # Convert from radians to degrees
        robot_angle = math.degrees(self.robots[robot][2]) % 360

        # Find the angle between the robot and the ball
        angle_to_ball = math.degrees(
            math.atan2(
                self.ball[1] - self.robots[robot][1],
                self.ball[0] - self.robots[robot][0],
            )
        )

        # Check if the robot is facing the ball
        angle = (robot_angle - angle_to_ball) % 360

        if angle < req_angle or angle > 360 - req_angle:
            return True
        else:
            return False

    """
    Gets relative position of object to agent
    """

    def get_relative_observation(self, agent_loc, object_loc):
        # Get relative position of object to agent, returns x, y, angle
        # Agent loc is x, y, angle
        # Object loc is x, y

        # Add noise to object location
        object_loc = [
            object_loc[0] + np.random.normal(0, self.observation_noise),
            object_loc[1] + np.random.normal(0, self.observation_noise),
        ]

        # Get relative position of object to agent
        x = object_loc[0] - agent_loc[0]
        y = object_loc[1] - agent_loc[1]
        angle = np.arctan2(y, x) - agent_loc[2]

        # Rotate x, y by -agent angle
        xprime = x * np.cos(-agent_loc[2]) - y * np.sin(-agent_loc[2])
        yprime = x * np.sin(-agent_loc[2]) + y * np.cos(-agent_loc[2])

        return [xprime / 10000, yprime / 10000, np.sin(angle), np.cos(angle)]
        # return [xprime / 10000, yprime / 10000]

    def kick_ball(self, robot, kick_strength):
        if self.check_facing_ball(robot):
            robot_location = np.array([self.robots[robot][0], self.robots[robot][1]])

            # Find distance between robot and ball
            ball_location = np.array([self.ball[0], self.ball[1]])

            distance_robot_ball = np.linalg.norm(ball_location - robot_location)

            # If robot is close enough to ball, kick ball
            if distance_robot_ball < self.robot_x_size:
                # normalize kick strength between 0.5
                
                # 0.6 to 1
                self.ball[2] = 70 * kick_strength + 20
# 
                # # Set ball direction to be robot angle + noise
                self.ball[3] = self.robots[robot][2]

                # set ball to be in direction of the agent to the ball
                # self.ball[3] = math.atan2(self.ball[1] - robot_location[1], self.ball[0] - robot_location[0])

                # self.ball_direction = math.atan2(self.ball[1] - robot_location[1], self.ball[0] - robot_location[0])

    ############ RENDERING UTILS ############

    def render_robot(self, robot):
        render_length = 1200
        render_robot_x = int((self.robots[robot][0] / 5200 + 1) * (render_length / 2))
        render_robot_y = int((self.robots[robot][1] / 3700 + 1) * (render_length / 3))
        
        # Draw robot direction
        pygame.draw.line(
            self.field,
            pygame.Color(50, 50, 50),
            (render_robot_x, render_robot_y),
            (
                render_robot_x + self.robot_x_size/20 * np.cos(self.robots[robot][2]),
                render_robot_y + self.robot_x_size/20 * np.sin(self.robots[robot][2]),
            ),
            width=5,
        )
        # Add robot number
        font = pygame.font.SysFont("Arial", 20)
        text = font.render(str(robot), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (render_robot_x, render_robot_y)
        self.field.blit(text, textRect)

        # Color = dark red
        color = (140, 0, 0)
        
        # Create a new surface with the size of the robot
        robot_surface = pygame.Surface((self.robot_x_size / 10, self.robot_y_size / 10), pygame.SRCALPHA)

        # Draw a rectangle on the new surface
        pygame.draw.rect(
            robot_surface,
            pygame.Color(color[0], color[1], color[2]),
            pygame.Rect(0, 0, self.robot_x_size / 10, self.robot_y_size / 10),
            width=5,
        )

        # Create a new surface that's centered on the original surface
        centered_surface = pygame.Surface((self.robot_x_size / 5, self.robot_y_size / 5), pygame.SRCALPHA)
        centered_surface.blit(robot_surface, (self.robot_x_size / 20, self.robot_y_size / 20))

        # Rotate the surface
        rotated_surface = pygame.transform.rotate(centered_surface, -self.robots[robot][2] * 180 / np.pi)

        # Calculate the position of the rotated surface
        render_robot_x -= rotated_surface.get_width() / 2
        render_robot_y -= rotated_surface.get_height() / 2

        # Draw the rotated surface on the field
        self.field.blit(rotated_surface, (render_robot_x, render_robot_y))

    """
    Field dimensions are on page 2:
    https://spl.robocup.org/wp-content/uploads/SPL-Rules-2023.pdf
    """
    def basic_field(self, _render_length=1200):
        render_length = _render_length

        # All dimensions are in mm proportional to the render_length
        # you can't change (based on the official robocup rule book ratio)

        # Total render length should be Border_strip_width * 2 + Field_length
        # Total render width should be Border_strip_width * 2 + Field_width

        # Field dimensions
        Field_length = 9000
        Field_width = 6000
        Line_width = 50
        Penalty_mark_size = 100
        Goal_area_length = 600
        Goal_area_width = 2200
        Goal_length = 500
        Goal_width = 1500
        Penalty_area_length = 1650
        Penalty_area_width = 4000
        Penalty_mark_distance = 1300
        Center_circle_diameter = 1500
        Border_strip_width = 700

        # Create render dimensions
        Field_length_render = (
            Field_length * render_length / (Field_length + 2 * Border_strip_width)
        )
        Field_width_render = (
            Field_width * render_length / (Field_length + 2 * Border_strip_width)
        )
        Line_width_render = int(
            Line_width * render_length / (Field_length + 2 * Border_strip_width)
        )
        Penalty_mark_size_render = (
            Penalty_mark_size * render_length / (Field_length + 2 * Border_strip_width)
        )
        Goal_area_length_render = (
            Goal_area_length * render_length / (Field_length + 2 * Border_strip_width)
        )
        Goal_area_width_render = (
            Goal_area_width * render_length / (Field_length + 2 * Border_strip_width)
        )
        Goal_length_render = (
            Goal_length * render_length / (Field_length + 2 * Border_strip_width)
        )
        Goal_width_render = (
            Goal_width * render_length / (Field_length + 2 * Border_strip_width)
        )
        Penalty_area_length_render = (
            Penalty_area_length
            * render_length
            / (Field_length + 2 * Border_strip_width)
        )
        Penalty_area_width_render = (
            Penalty_area_width * render_length / (Field_length + 2 * Border_strip_width)
        )
        Penalty_mark_distance_render = (
            Penalty_mark_distance
            * render_length
            / (Field_length + 2 * Border_strip_width)
        )
        Center_circle_diameter_render = (
            Center_circle_diameter
            * render_length
            / (Field_length + 2 * Border_strip_width)
        )
        Border_strip_width_render = int(
            Border_strip_width * render_length / (Field_length + 2 * Border_strip_width)
        )
        Surface_width = int(Field_length_render + 2 * Border_strip_width_render)
        Surface_height = (
            int(Field_width_render + 2 * Border_strip_width_render) - 40
        )  # Constant here is just to make it look correct, unsure why it is needed

        Soccer_green = (18, 160, 0)
        self.field.fill(Soccer_green)

        # Draw out of bounds lines
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (Border_strip_width_render, Border_strip_width_render),
            (Surface_width - Border_strip_width_render, Border_strip_width_render),
            width=Line_width_render,
        )
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (Border_strip_width_render, Surface_height - Border_strip_width_render),
            (
                Surface_width - Border_strip_width_render,
                Surface_height - Border_strip_width_render,
            ),
            width=Line_width_render,
        )
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (Border_strip_width_render, Border_strip_width_render),
            (Border_strip_width_render, Surface_height - Border_strip_width_render),
            width=Line_width_render,
        )
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (Surface_width - Border_strip_width_render, Border_strip_width_render),
            (
                Surface_width - Border_strip_width_render,
                Surface_height - Border_strip_width_render,
            ),
            width=Line_width_render,
        )

        # Draw center line
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (Surface_width / 2, Border_strip_width_render),
            (Surface_width / 2, Surface_height - Border_strip_width_render),
            width=Line_width_render,
        )

        # Draw center circle
        pygame.draw.circle(
            self.field,
            pygame.Color(255, 255, 255),
            (int(Surface_width / 2), int(Surface_height / 2)),
            int(Center_circle_diameter_render / 2),
            width=Line_width_render,
        )

        # Draw center dot
        pygame.draw.circle(
            self.field,
            pygame.Color(255, 255, 255),
            (int(Surface_width / 2), int(Surface_height / 2)),
            int(Line_width_render / 2),
        )

        # Draw penalty areas
        # Left penalty area. Should be 1650mm long and 4000mm wide starting at goal line
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (
                Border_strip_width_render,
                Surface_height / 2 - Penalty_area_width_render / 2,
            ),
            (
                Border_strip_width_render + Penalty_area_length_render,
                Surface_height / 2 - Penalty_area_width_render / 2,
            ),
            width=Line_width_render,
        )
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (
                Border_strip_width_render,
                Surface_height / 2 + Penalty_area_width_render / 2,
            ),
            (
                Border_strip_width_render + Penalty_area_length_render,
                Surface_height / 2 + Penalty_area_width_render / 2,
            ),
            width=Line_width_render,
        )
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (
                Border_strip_width_render + Penalty_area_length_render,
                Surface_height / 2 - Penalty_area_width_render / 2,
            ),
            (
                Border_strip_width_render + Penalty_area_length_render,
                Surface_height / 2 + Penalty_area_width_render / 2,
            ),
            width=Line_width_render,
        )

        # Right penalty area. Should be 1650mm long and 4000mm wide starting at goal line
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (
                Surface_width - Border_strip_width_render,
                Surface_height / 2 - Penalty_area_width_render / 2,
            ),
            (
                Surface_width - Border_strip_width_render - Penalty_area_length_render,
                Surface_height / 2 - Penalty_area_width_render / 2,
            ),
            width=Line_width_render,
        )
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (
                Surface_width - Border_strip_width_render,
                Surface_height / 2 + Penalty_area_width_render / 2,
            ),
            (
                Surface_width - Border_strip_width_render - Penalty_area_length_render,
                Surface_height / 2 + Penalty_area_width_render / 2,
            ),
            width=Line_width_render,
        )
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (
                Surface_width - Border_strip_width_render - Penalty_area_length_render,
                Surface_height / 2 - Penalty_area_width_render / 2,
            ),
            (
                Surface_width - Border_strip_width_render - Penalty_area_length_render,
                Surface_height / 2 + Penalty_area_width_render / 2,
            ),
            width=Line_width_render,
        )

        # Draw goal areas
        # Left goal area. Should be 600mm long and 2200mm wide starting at goal line
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (
                Border_strip_width_render,
                Surface_height / 2 - Goal_area_width_render / 2,
            ),
            (
                Border_strip_width_render + Goal_area_length_render,
                Surface_height / 2 - Goal_area_width_render / 2,
            ),
            width=Line_width_render,
        )
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (
                Border_strip_width_render,
                Surface_height / 2 + Goal_area_width_render / 2,
            ),
            (
                Border_strip_width_render + Goal_area_length_render,
                Surface_height / 2 + Goal_area_width_render / 2,
            ),
            width=Line_width_render,
        )
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (
                Border_strip_width_render + Goal_area_length_render,
                Surface_height / 2 - Goal_area_width_render / 2,
            ),
            (
                Border_strip_width_render + Goal_area_length_render,
                Surface_height / 2 + Goal_area_width_render / 2,
            ),
            width=Line_width_render,
        )

        # Right goal area. Should be 600mm long and 2200mm wide starting at goal line
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (
                Surface_width - Border_strip_width_render,
                Surface_height / 2 - Goal_area_width_render / 2,
            ),
            (
                Surface_width - Border_strip_width_render - Goal_area_length_render,
                Surface_height / 2 - Goal_area_width_render / 2,
            ),
            width=Line_width_render,
        )
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (
                Surface_width - Border_strip_width_render,
                Surface_height / 2 + Goal_area_width_render / 2,
            ),
            (
                Surface_width - Border_strip_width_render - Goal_area_length_render,
                Surface_height / 2 + Goal_area_width_render / 2,
            ),
            width=Line_width_render,
        )
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (
                Surface_width - Border_strip_width_render - Goal_area_length_render,
                Surface_height / 2 - Goal_area_width_render / 2,
            ),
            (
                Surface_width - Border_strip_width_render - Goal_area_length_render,
                Surface_height / 2 + Goal_area_width_render / 2,
            ),
            width=Line_width_render,
        )

        # Draw penalty marks
        # Left penalty mark. Should be 100mm in diameter and 1300mm from goal line
        pygame.draw.circle(
            self.field,
            pygame.Color(255, 255, 255),
            (
                Border_strip_width_render + Penalty_mark_distance_render,
                Surface_height / 2,
            ),
            int(Penalty_mark_size_render / 2),
            width=Line_width_render,
        )

        # Right penalty mark. Should be 100mm in diameter and 1300mm from goal line
        pygame.draw.circle(
            self.field,
            pygame.Color(255, 255, 255),
            (
                Surface_width
                - Border_strip_width_render
                - Penalty_mark_distance_render,
                Surface_height / 2,
            ),
            int(Penalty_mark_size_render / 2),
            width=Line_width_render,
        )

        # Draw center point, same size as penalty mark
        pygame.draw.circle(
            self.field,
            pygame.Color(255, 255, 255),
            (Surface_width / 2, Surface_height / 2),
            int(Penalty_mark_size_render / 2),
            width=Line_width_render,
        )

        # Fill in goal areas with light grey, should mirror goal area flipped along goal line
        # Left goal area
        pygame.draw.rect(
            self.field,
            pygame.Color(255, 153, 153),
            (
                Border_strip_width_render - Goal_length_render,
                Surface_height / 2 - Goal_width_render / 2 - Line_width_render / 2,
                Goal_length_render,
                Goal_width_render,
            ),
        )

        # TODO: Make goal areas look better
        # Draw lines around goal areas
        # Left goal area
        # pygame.draw.line(
        #     self.field,
        #     pygame.Color(255, 255, 255),
        #     (Border_strip_width_render - Goal_area_length_render, Surface_height / 2 - Goal_area_width_render / 2),
        #     (Border_strip_width_render, Surface_height / 2 - Goal_area_width_render / 2),
        #     width=Line_width_render,
        # )
        # pygame.draw.line(
        #     self.field,
        #     pygame.Color(255, 255, 255),
        #     (Border_strip_width_render - Goal_area_length_render, Surface_height / 2 + Goal_area_width_render / 2),
        #     (Border_strip_width_render, Surface_height / 2 + Goal_area_width_render / 2),
        #     width=Line_width_render,
        # )

        # Right goal area
        pygame.draw.rect(
            self.field,
            pygame.Color(153, 204, 255),
            (
                Surface_width - Border_strip_width_render,
                Surface_height / 2 - Goal_width_render / 2 - Line_width_render / 2,
                Goal_length_render,
                Goal_width_render,
            ),
        )

    def render(self, mode="human"):
        render_length = 1200
        time.sleep(0.01)

        Field_length = 9000
        Field_width = 6000
        Border_strip_width = 700

        Field_length_render = (
            Field_length * render_length / (Field_length + 2 * Border_strip_width)
        )
        Field_width_render = (
            Field_width * render_length / (Field_length + 2 * Border_strip_width)
        )
        Border_strip_width_render = (
            Border_strip_width * render_length / (Field_length + 2 * Border_strip_width)
        )
        Surface_width = int(Field_length_render + 2 * Border_strip_width_render)
        Surface_height = int(Field_width_render + 2 * Border_strip_width_render)

        if self.rendering_init == False:
            pygame.init()

            self.field = pygame.display.set_mode((Surface_width, Surface_height))

            self.basic_field(render_length)
            pygame.display.set_caption("Point Targeting Environment")
            self.clock = pygame.time.Clock()

            self.rendering_init = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # If space on keyboard is pressed, reset the environment
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            self.reset()

        self.basic_field(render_length)

        # Render robots
        for robot in self.agents:
            self.render_robot(robot)

        # for i in range(self.num_defenders):
        #     self.render_defender(i)

        # Render ball
        render_ball_x = int((self.ball[0] / 5200 + 1) * (render_length / 2))
        render_ball_y = int((self.ball[1] / 3700 + 1) * (render_length / 3))

        pygame.draw.circle(
            self.field,
            pygame.Color(40, 40, 40),
            (render_ball_x, render_ball_y),
            self.ball_radius,
        )

        pygame.display.update()
        self.clock.tick(60)


#########################################


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # Return true if line segments AB and CD intersect, for goal line
    def intersect(A, B, C, D):
        def ccw(A, B, C):
            return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)

        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


class History:
    def __init__(self, max_length):
        self.history = []
        self.max_length = max_length

    def add(self, item):
        self.history.append(item)
        if len(self.history) > self.max_length:
            self.history.pop(0)

    def get(self):
        return self.history

    def getMaxLength(self):
        return self.max_length
