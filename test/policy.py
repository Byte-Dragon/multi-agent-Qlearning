import numpy as np
from pyglet.window import key
import random
# individual agent policy
class Policy(object):
    def __init__(self):
        pass
    def action(self, obs):
        raise NotImplementedError()

# interactive policy based on keyboard input
# hard-coded to deal only with movement, not communication
class InteractivePolicy(Policy):
    def __init__(self, env, agent_index):
        super(InteractivePolicy, self).__init__()
        self.env = env
        self.agent = self.env.world.policy_agents[agent_index]
        # hard-coded keyboard events
        self.move = [False for i in range(4)]
        self.comm = [False for i in range(env.world.dim_c)]
        # register keyboard events with this environment's window
        # env.viewers[agent_index].window.on_key_press = self.key_press
        # env.viewers[agent_index].window.on_key_release = self.key_release

    def action(self, obs):
        # ignore observation and just act based on keyboard events
        k = random.choice([key.LEFT,key.RIGHT,key.UP,key.DOWN])
        self.key_press(k)
        if self.env.discrete_action_input:
            u = 0
            if self.move[0]: u = 1
            if self.move[1]: u = 2
            if self.move[2]: u = 4
            if self.move[3]: u = 3
        else:
            u = np.zeros(self.env.world.dim_p*2 + 1)  # 5-d because of no-move action
            c = np.zeros(self.env.world.dim_c)  # 5-d because of no-move action
            if self.move[0]:
                u[1] += 1.0
            if self.move[1]:
                u[2] += 1.0
            if self.move[3]:
                u[3] += 1.0
            if self.move[2]:
                u[4] += 1.0
            if True not in self.move:
                u[0] += 1.0
        c = [self.env.get_reward(self.agent)]
        self.key_release(k)
        return np.concatenate([u, c])

    # keyboard event callbacks
    def key_press(self, k):
        if k==key.LEFT:  self.move[0] = True
        if k==key.RIGHT: self.move[1] = True
        if k==key.UP:    self.move[2] = True
        if k==key.DOWN:  self.move[3] = True

    def key_release(self, k):
        if k==key.LEFT:  self.move[0] = False
        if k==key.RIGHT: self.move[1] = False
        if k==key.UP:    self.move[2] = False
        if k==key.DOWN:  self.move[3] = False
