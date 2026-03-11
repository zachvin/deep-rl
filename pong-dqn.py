# agent
#   target network
#   q network

# data classes

import typing as tt
import numpy as np
import torch
import collections
import gymnasium as gym

BatchTensors = tt.Tuple[
    torch.ByteTensor, # state
    torch.LongTensor, # action (why use long?)
    torch.Tensor, # reward
    torch.BoolTensor, # done/trunc
    torch.ByteTensor # next state (what is byte tensor?)
]

@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    done_trunc = bool
    new_state = np.ndarray

class ExperienceBuffer:
    def __init__(self, buffer_len):
        self.buffer = collections.deque(maxlen=buffer_len)

    def __len__(self):
        return len(self.buffer)

    def append(self, e: Experience):
        self.buffer.append(e)

    def sample(self, n: int) -> tt.List[Experience]:
        # randomly sample N experiences
        indices = np.random.choice(len(self), n, replace=False)
        return [self.buffer[idx] for idx in indices]

class Agent:
    # init, reset, play_step
    def __init__(self, env: gym.Env, buffer: ExperienceBuffer):
        self.env = env
        self.buffer = buffer
        self.state = tt.Optional[np.ndarray] = None # why tt.Optional?
        self._reset()

    def _reset(self):
        self.state, _ = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net: dqn_model.DQN, device: torch.device,
                  epsilon: float = 0.0) -> tt.Optional[float]: done_reward = None
        
        if np.random() > epsilon:
            action = self.action_space.sample()
        else:
            state = torch.as_tensor(self.state).to(device)
            state.unsqueeze_(0) # unsqueeze_ to do inplace. Torch expects extra batch dimension
            q_vals = net(state)
            _, act = torch.max(q_vals, dim=1) # max value action 
            action = int(act.item()) # gymnasium expects non-tensor, integer action

        new_state, reward, done, trunc, _ = self.env.step(action)
        self.total_reward += reward # record this why? maybe just for our metrics?

        exp = Experience(
            state=self.state, action=action, reward=float(reward), # reward must be float type, I guess isn't already?
            new_state=new_state, done_trunc=done or trunc
        )
        self.buffer.append(exp)
        self.state = new_state # update for next step
        if done or trunc:
            done_reward = self.total_reward
            self._reset()

        return done_reward

def main():
    # make gym env
    # loop
    #   choose action
    #   use action
    #   record memory
    #   if enough memories, learn