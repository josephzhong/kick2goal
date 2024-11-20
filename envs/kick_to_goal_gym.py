from __future__ import annotations
import copy
import functools
import time
from typing import Dict, List, Optional, Tuple, TypeVar, Union, cast

from numpy.f2py.auxfuncs import throw_error
from overrides import overrides
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import AgentID, ObsType, ActionType
import gymnasium as gym
import pygame
import math
import numpy as np
import random
import sys

sys.path.append(sys.path[0] + "/..")


def env():
    env = KickToGoalGym()
    return env


class KickToGoalGym(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode="rgb_array", seed=None, episode_length=1000, goal_reward=100):
        self.rendering_init = False
        self.render_mode = render_mode
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

        self.reward_dict = {
            "ball_to_goal": 0.1,
            "agent_to_ball" : 0.01,
            "looking_at_ball": 0.001,
            "kick": 1,
            "missed_kick": -1,
            "goal": goal_reward
        }

        # self.reward_dict = {
        #     "ball_to_goal": 0.1,
        #     "agent_to_ball": 0.0,
        #     "looking_at_ball": 0.001,
        #     "kick": 1,
        #     "missed_kick": -1,
        #     "goal": 1000
        # }


        # self.reward_dict = {
        #     "ball_to_goal": 0.0,
        #     "agent_to_ball": 0.0,
        #     "looking_at_ball": 0.0,
        #     "kick": 0,
        #     "missed_kick": 0,
        #     "goal": 1000,
        #     "out_of_bounds": -1000
        # }

        # Parameters
        self.episode_length = episode_length

        self.ball_acceleration = -0.8
        self.ball_velocity_coef = 3
        self.last_touched = None

        self.displacement_coef = 0.15
        self.angle_displacement = 0.15
        
        self.robot_x_size = 200
        self.robot_y_size = 350
        self.ball_radius = 10

        self.num_defenders = 0
        self.defenders = []
        self.defender_info = []

        # agents
        self.num_robots = 1

        self.possible_agents = list(range(self.num_robots))
        self.possible_agents = cast(List[AgentID], self.possible_agents)
        self.agents: List[AgentID] = self.possible_agents[:]

        # 4D vector of (x, y, angle, kick_strength) velocity changes
        self.action_space = gym.spaces.Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
        )

        # observation spaces
        history_length = 3
        self.obs_history = {agent: History(history_length) for agent in self.agents}

        self.reset_multi_agents()
        obs_size = len(self.get_obs(0, include_history=True))

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(obs_size,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID = 0) -> gym.spaces.Space:
        return self.observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID = 0) -> gym.spaces.Space:
        return self.action_space

    def reset_seed(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.reset()

    def update_attribute(self, attribute_name, value):
        assert hasattr(self, attribute_name)
        setattr(self, attribute_name, value)
        print(f"env attribute {attribute_name} changed to {value}")

    def get_distance(self, pos1: List[float], pos2: List[float]) -> float:
        return np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))

    """
    Takes as input robot or opponent robot NAME
    """

    def can_kick(self, robot: AgentID = 0) -> bool:
        robot_loc = self.robots[robot]

        if self.check_facing_ball(robot):
            return (
                self.get_distance(robot_loc, self.ball)
                < self.robot_x_size
            )

    """
    ego-centric observation:
        origin,
        goal,
        other robots,
        ball
    """

    def get_obs(self, robot: AgentID = 0, include_history=False) -> List[float]:
        robot_loc = self.robots[robot]
        obs: List[float] = []

        # Ball
        ball = self.get_relative_observation(robot_loc, self.ball)
        obs.extend(ball)

        # Goal
        goal = self.get_relative_observation(robot_loc, [4800, 0])
        obs.extend(goal)

        # Opponent goal
        opp_goal = self.get_relative_observation(robot_loc, [-4800, 0])
        obs.extend(opp_goal)

        if include_history:
            for prev_obs in self.obs_history[robot].get():
                obs.extend(prev_obs)

        return np.array(obs, dtype=np.float32)

    """
    Format for robots:
    x, y, angle

    Format for robot velocities:
    x, y, angle

    Format for ball:
    x, y, velocity, angle
    """

    @overrides
    def reset(self,
              seed: Union[int, None] = None,
              options: Union[dict, None] = None,
        ):
        r = self.reset_multi_agents()
        return r[0][0], r[1][0]

    def reset_multi_agents(
        self,
        seed: Union[int, None] = None,
        options: Union[dict, None] = None,
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        self.time = 0
        self.terminated_dict = {
            "goal_scored": False,
            "out_of_bounds": False,
        }

        # Defenders
        self.robots: Dict[AgentID, List[float]] = {
            0: [self.rng.uniform(-4000, 4000), self.rng.uniform(-3000, 3000), 0],
        }
        self.ball: List[float] = [self.rng.uniform(-4000, 4000), self.rng.uniform(-3000, 3000), 0, 0]

        self.robot_velocities = [[0, 0, 0] for _ in range(self.num_robots)]

        for agent in self.agents:
            for _ in range(self.obs_history[agent].getMaxLength()):
                self.obs_history[agent].add(self.get_obs(agent, include_history=False))

        observations = {}
        for agent in self.agents:
            observations[agent] = self.get_obs(agent, include_history=True)

        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, action: ActionType):
        r = self.step_multi_agents({0: action})
        infor = {"goal": 1 if self.terminated_dict["goal_scored"] else 0}
        return r[0][0], r[1][0], r[2][0], r[3][0], infor

    def step_multi_agents(
        self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, ObsType], # observation dict
        dict[AgentID, float], # reward dict
        dict[AgentID, bool], # terminated dict
        dict[AgentID, bool], # truncated dict
        dict[AgentID, dict], # info dict
    ]:
        """
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
        self.prev_agents = copy.deepcopy(self.robots)
        self.prev_ball = copy.deepcopy(self.ball)

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

        if self.terminated_dict["goal_scored"] or self.terminated_dict["out_of_bounds"]:
            terminated = {agent: True for agent in self.agents}

        for agent in self.agents:
            self.obs_history[agent].add(self.get_obs(agent, include_history=False))

        return obs, rew, terminated, truncated, info

    """
    Checks if ball is in goal area
    """

    def goal(self) -> bool:
        # Note: This is tighter than the actual goal for transfer performance
        if self.ball[0] > 4500 and self.ball[1] < 300 and self.ball[1] > -300:
            return True
        return False

    """
    Get angle for contacting other robot
    """

    def get_angle(self, pos1: List[float], pos2: List[float]) -> float:
        return np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0])

    def calculate_reward(self, agent: AgentID, action: ActionType) -> float:
        reward = 0

        # Looking at ball
        if self.check_facing_ball(agent, req_angle=5):
            reward += self.reward_dict["looking_at_ball"]

        # Agent to ball
        reward += self.reward_dict["agent_to_ball"] * (
            self.get_distance(self.prev_agents[agent], self.prev_ball)
            - self.get_distance(self.robots[agent], self.ball)
        )

        # Ball to goal - Team
        reward += self.reward_dict["ball_to_goal"] * (
            self.get_distance(self.prev_ball, [4800, 0])
            - self.get_distance(self.ball, [4800, 0])
        )
        # Ball to goal - Team
        if self.goal():
            self.terminated_dict["goal_scored"] = True
            reward += self.reward_dict["goal"]

            # if random.random() < 0.1:
            # print(self.reward_dict)
            # print(reward)
        # add out of boundary penalty
        # else:
        #     if self.terminated_dict["out_of_bounds"]:
        #         reward += self.reward_dict["out_of_bounds"]

        if action[3] > 0.2:
            # Kick
            if self.can_kick(agent):
                reward += self.reward_dict["kick"]
            else:
                reward += self.reward_dict["missed_kick"]

        return reward
    
    def robot_out_of_bounds(self, robot: AgentID) -> bool:
        if self.robots[robot][0] > 5200 or self.robots[robot][0] < -5200:
            return True
        if self.robots[robot][1] > 3700 or self.robots[robot][1] < -3700:
            return True
        return False

    def check_defender_collision(self, robot: AgentID, defender: AgentID) -> bool:
        robot_location = np.array([self.robots[robot][0], self.robots[robot][1]])
        defender_location = np.array(
            [self.defenders[defender][0], self.defenders[defender][1]]
        )

        distance_robots = np.linalg.norm(defender_location - robot_location)

        # If collision, adjust velocities to bouce off each other
        if distance_robots < (self.robot_radius + self.robot_radius) * 8:
            return True

    #### STEP UTILS ####
    def clip_velocities(self, velocities: ActionType) -> ActionType:
        for v in range(len(velocities)):
            # creates deadzone in middle of action space for sim2real transfer
            if abs(velocities[v]) < 0.2:
                velocities[v] = 0
            else:
                velocities[v] = np.sign(velocities[v]) * (abs(velocities[v]) - 0.2)

        clips = [
            [1, -0.3],
            [0.5, -0.5],
            [1, -1],
        ]
        # clip
        for i in range(len(clips)):
            velocities[i] = np.clip(velocities[i], clips[i][1], clips[i][0])

        return velocities

    def move_agent(self, robot: AgentID, action: ActionType) -> None:
        if action[3] > 0.2:
            # Kick the ball
            self.kick_ball(robot, action[3])
        else:
            # Clip velocities
            clipped_action = self.clip_velocities(
                action
            )

            policy_goal_x = self.robots[robot][0] + (
                (
                    (np.cos(self.robots[robot][2]) * clipped_action[0])
                    + (
                        np.cos(self.robots[robot][2] + np.pi / 2)
                        * clipped_action[1]
                    )
                )
                * 100
            )  # the x component of the location targeted by the high level action
            policy_goal_y = self.robots[robot][1] + (
                (
                    (np.sin(self.robots[robot][2]) * clipped_action[0])
                    + (
                        np.sin(self.robots[robot][2] + np.pi / 2)
                        * clipped_action[1]
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
                + clipped_action[2] * self.angle_displacement
            )

        # Make sure robot is on field
        self.robots[robot][0] = np.clip(self.robots[robot][0], -5200, 5200)
        self.robots[robot][1] = np.clip(self.robots[robot][1], -3700, 3700)

    def check_collision_ball(self, robot: AgentID) -> Tuple[bool, Optional[float]]:
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
    
    # NOTE: This doesn't work yet. It is just here for reference (for future use)
    def check_collision_robots(self, robot1, robot2) -> bool:
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
    
    def update_ball(self) -> None:
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
                self.ball[3] = angle


        # If ball OOB, terminate
        if abs(self.ball[0]) > 4500 or abs(self.ball[1]) > 3000:
            self.terminated_dict["out_of_bounds"] = True


    def check_facing_ball(self, robot: List[float], req_angle=10) -> bool:
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

    def get_relative_observation(self, agent_loc: List[float], object_loc: List[float]) -> List[float]:
        # Get relative position of object to agent, returns x, y, angle
        # Agent loc is x, y, angle
        # Object loc is x, y

        # Get relative position of object to agent
        x = object_loc[0] - agent_loc[0]
        y = object_loc[1] - agent_loc[1]
        angle = np.arctan2(y, x) - agent_loc[2]

        # Rotate x, y by -agent angle
        xprime = x * np.cos(-agent_loc[2]) - y * np.sin(-agent_loc[2])
        yprime = x * np.sin(-agent_loc[2]) + y * np.cos(-agent_loc[2])

        return [xprime / 10000, yprime / 10000, np.sin(angle), np.cos(angle)]

    def kick_ball(self, robot: AgentID, kick_strength: float) -> None:
        if self.check_facing_ball(robot):
            robot_location = np.array([self.robots[robot][0], self.robots[robot][1]])

            # Find distance between robot and ball
            ball_location = np.array([self.ball[0], self.ball[1]])

            distance_robot_ball = np.linalg.norm(ball_location - robot_location)

            # If robot is close enough to ball, kick ball
            if distance_robot_ball < self.robot_x_size * 2:
                self.ball[2] = 60 * kick_strength + 10

                self.ball[3] = self.robots[robot][2]

    ############ RENDERING UTILS ############

    def render_robot(self, robot: AgentID) -> None:
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

    # NOTE: Defenders are just here for future reference. Currently code is not used and they are circles
    # def render_defender(self, defender: AgentID) -> None:
    #     # Pink = 255 51 255
    #     # Blue = 63 154 246

    #     render_length = 1200
    #     render_robot_x = int(
    #         (self.defenders[defender][0] / 5200 + 1) * (render_length / 2)
    #     )
    #     render_robot_y = int(
    #         (self.defenders[defender][1] / 3700 + 1) * (render_length / 3)
    #     )

    #     # Color = dark red
    #     color = (63, 154, 246)

    #     # Draw robot
    #     pygame.draw.circle(
    #         self.field,
    #         pygame.Color(color[0], color[1], color[2]),
    #         (render_robot_x, render_robot_y),
    #         self.robot_radius,
    #         width=5,
    #     )

    #     # # Draw robot direction
    #     # pygame.draw.line(
    #     #     self.field,
    #     #     pygame.Color(50, 50, 50),
    #     #     (render_robot_x, render_robot_y),
    #     #     (
    #     #         render_robot_x
    #     #         + self.robot_radius * np.cos(self.defenders[defender][2]),
    #     #         render_robot_y
    #     #         + self.robot_radius * np.sin(self.defenders[defender][2]),
    #     #     ),
    #     #     width=5,
    #     # )
    #     # Add robot number
    #     font = pygame.font.SysFont("Arial", 20)
    #     text = font.render(str(defender), True, (0, 0, 0))
    #     textRect = text.get_rect()
    #     textRect.center = (render_robot_x, render_robot_y)
    #     self.field.blit(text, textRect)

    """
    Field dimensions are on page 2:
    https://spl.robocup.org/wp-content/uploads/SPL-Rules-2023.pdf
    """

    def basic_field(self, _render_length=1200) -> None:
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
                Border_strip_width_render - Goal_area_length_render,
                Surface_height / 2 - Goal_area_width_render / 2 - Line_width_render / 2,
                Goal_area_length_render,
                Goal_area_width_render,
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
        # pygame.draw.rect(
        #     self.field,
        #     pygame.Color(153, 204, 255),
        #     (
        #         Surface_width - Border_strip_width_render,
        #         Surface_height / 2 - Goal_area_width_render / 2 - Line_width_render / 2,
        #         Goal_area_length_render,
        #         Goal_area_width_render,
        #     ),
        # )
        # tight condition for goal
        tight_width = 600
        Goal_area_width_render = (
                tight_width * render_length / (Field_length + 2 * Border_strip_width)
        )
        pygame.draw.rect(
            self.field,
            pygame.Color(153, 204, 255),
            (
                Surface_width - Border_strip_width_render,
                Surface_height / 2 - Goal_area_width_render / 2 - Line_width_render / 2,
                Goal_area_length_render,
                Goal_area_width_render,
            ),
        )

    @overrides
    def render(self) -> Union[None, np.ndarray, str, list]:
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

        self.basic_field(render_length)

        # Render robots
        for robot in self.agents:
            self.render_robot(robot)

        for i in range(self.num_defenders):
            self.render_defender(i)

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

    def close(self):
        if self.rendering_init:
            pygame.display.quit()
        super().close()



#########################################


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    # Return true if line segments AB and CD intersect, for goal line
    def intersect(A: Point, B: Point, C: Point, D: Point) -> bool:
        def ccw(A, B, C):
            return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)

        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

T = TypeVar("T")

class History:
    def __init__(self, max_length: int):
        self.history: List[T] = []
        self.max_length = max_length

    def add(self, item: T):
        self.history.append(item)
        if len(self.history) > self.max_length:
            self.history.pop(0)

    def get(self) -> List[T]:
        return self.history

    def getMaxLength(self) -> int:
        return self.max_length
