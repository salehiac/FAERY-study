import torch
import random
import numpy as np


# use __slots__
# dict holding all attributes, makes creation of many instances much faster
# currently not working with deap's hall_of_fame.. needs debugging
class GridAgent:
    """
    A grid agent, able to take an action in GridWorld
    """

    id = 0
    
    # __slots__ = ("state_hist", "behavior", "action")
    def __init__(self):
        super().__init__()

        self.id = GridAgent.id
        GridAgent.id += 1
        
        # Behavioral descriptor of the agent
        self.state_hist = []
        self.behavior = None
        self.action = None
    
    def update_behavior(self, state_hist):
        """
        Updates the agent's behavior descriptor based on its trajectory
        """

        self.state_hist = state_hist
        self.behavior = self.state_hist[-1]
        return self.behavior
    

class GridAgentNN(GridAgent, torch.nn.Module):
    """
    A grid agent, able to take an action with a neural network
    """

    # __slots__ = ("mds", "non_linearity", "batchnorm", "output_normaliser", "weights")
    def __init__(
        self,

        input_dim=2, output_dim=2,
        hidden_layers=3, hidden_dim=10,
        use_batchnorm=False,
        non_linearity=torch.tanh, 
        output_normalizer=lambda x: x,
    ):
        """
        Enter NN parameters aswell as agent's lineage
        """

        GridAgent.__init__(self)
        torch.nn.Module.__init__(self)

        # Agent's network
        self.mds = torch.nn.ModuleList([torch.nn.Linear(input_dim, hidden_dim)])

        for i in range(hidden_layers - 1):
            self.mds.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.mds.append(torch.nn.Linear(hidden_dim, output_dim))

        self.non_linearity = non_linearity
        self.batchnorm = torch.nn.BatchNorm1d(hidden_dim) if use_batchnorm else lambda x: x

        self.output_normaliser = output_normalizer

        self.weights = self.get_flattened_weights()
    
    def forward(self, x, return_type="tuple"):
        """
        x : list
        return type : tuple; numpy; torch
        """

        self.action = torch.Tensor(x).unsqueeze(0)
        self.action = self.non_linearity(self.mds[0](self.action))
        for md in self.mds[1:-1]:
            self.action = self.batchnorm(self.non_linearity(md(self.action)))

        self.action = self.non_linearity(self.mds[-1](self.action))
        self.action = self.output_normaliser(self.action)
        
        if return_type == "tuple":
            return tuple(*self.action.numpy())
        elif return_type == "numpy":
            return self.action.detach().cpu().numpy()
        else:
            return self.action

    def get_flattened_weights(self):
        """
        Returns the agent's network's weights as list
        """

        flattened = []
        for m in self.mds:
            flattened += m.weight.view(-1).tolist()
            flattened += m.bias.view(-1).tolist()

        return flattened

    def set_flattened_weights(self, w_in):
        """
        w_in list
        """

        with torch.no_grad():
            start = 0
            for m in self.mds:
                w, b = m.weight, m.bias

                num_w = np.prod(list(w.shape))
                num_b = np.prod(list(b.shape))

                m.weight.data = torch.Tensor(w_in[start: \
                    start + num_w]).reshape(w.shape)
                m.bias.data = torch.Tensor(w_in[start + num_w: \
                    start + num_w + num_b]).reshape(b.shape)

                start = start + num_w + num_b
        
        self.weights = self.get_flattened_weights()
        
    def __len__(self):
        return len(self.weights)
    
    def __getitem__(self, key):
        return self.weights[key]
    
    def __setitem__(self, key, value):
        new_weights = self.weights[:key] + [value] + self.weights[key+1:]
        self.set_flattened_weights(new_weights)


class GridAgentGuesser(GridAgent):
    """
    An agent that can take an action in guessing game only
    with special mutation operator
    """

    # __slots__ = ("grid_size", "action")
    def __init__(self, grid_size):
        super().__init__()

        self.grid_size = grid_size

        self.action = tuple([
            random.randint(0, self.grid_size-1),
            random.randint(0, self.grid_size-1)
        ])
    
    def __call__(self, *args, **kwds):
        return self.action
    
    def mutate(self):
        """
        A mutation is LEFT, RIGHT, UP, DOWN move
        """

        mutation = random.choice(["LEFT", "RIGHT", "UP", "DOWN"])
        if mutation == "LEFT":
            self.action = tuple([
                self.action[0],
                max(0, self.action[1] - 1)
            ])

        elif mutation == "RIGHT":
            self.action = tuple([
                self.action[0],
                min(self.grid_size - 1, self.action[1] + 1)
            ])

        elif mutation == "UP":
            self.action = tuple([
                max(0, self.action[0] - 1),
                self.action[1]
            ])

        elif mutation == "DOWN":
            self.action = tuple([
                min(self.grid_size - 1, self.action[0] + 1),
                self.action[1]
            ])
        
    def update_behavior(self, state_hist):
        """
        Updates the agent's behavior descriptor based on its trajectory
        Guesser takes one agent per game, so we're interested in its trajectory over all games
        """

        self.state_hist += state_hist
        self.behavior = self.state_hist[-1]
        return self.behavior


if __name__ == "__main__":

    from class_gridworld import GridWorld
    from utils_worlds import GridWorldSparse40x40Mixed
    from class_distribution_shapes import UniformRectangle
    
    GridWorldSparse40x40Mixed["start_distribution"] = UniformRectangle((0,0), 40, 40)
    g = GridWorld(**GridWorldSparse40x40Mixed, is_guessing_game=True)

    list_ag = [
        GridAgentNN(
            output_dim=2,
            output_normalizer=lambda x: torch.round(abs(g.size * x)).int()
        )
        for _ in range(5)
    ]
    for ag in list_ag:
        g(ag, 1)
    
    g.visualise_as_grid(
        list_state_hist=[ag.state_hist for ag in list_ag],
        show_start=False,
        show=True
    )