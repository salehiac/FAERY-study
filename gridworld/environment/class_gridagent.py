import torch
import random
import numpy as np

from perlin_noise import PerlinNoise


# use __slots__
# dict holding all attributes, makes creation of many instances much faster
# currently not working with deap's hall_of_fame.. needs debugging
class GridAgent:
    """
    A grid agent, able to take an action in GridWorld
    """

    id = 0
    
    # __slots__ = ("state_hist", "behavior", "action")
    def __init__(self, init_position=None):
        super().__init__()

        self.id = GridAgent.id
        GridAgent.id += 1
        
        # Behavioral descriptor of the agent
        self.state_hist = []
        self.update_behavior([init_position])
        self.action = init_position
        self.done = False

        if init_position is None:
            self.state_hist = []
    
    def __call__(self, *args, **kwds):
        self.action = self.behavior
    
    def update_behavior(self, state_hist):
        """
        Updates the agent's behavior descriptor based on its trajectory
        """

        self.state_hist += state_hist
        self.behavior = self.state_hist[-1]
        return self.behavior
    

class GridAgentNN(GridAgent, torch.nn.Module):
    """
    A grid agent, able to take an action with a neural network
    """

    # __slots__ = ("mds", "non_linearity", "batchnorm", "output_normaliser", "weights")
    def __init__(
        self,

        init_position=None,

        input_dim=2, output_dim=2,
        hidden_layers=3, hidden_dim=10,
        use_batchnorm=False,
        non_linearity=torch.tanh, 
        output_normalizer=lambda x: x,
    ):
        """
        Enter NN parameters aswell as agent's lineage
        """

        GridAgent.__init__(self, init_position=init_position)
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
        GridAgent.__call__(self)

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

    direction_list = ["LEFT", "RIGHT", "UP", "DOWN"]
    evo_map = None

    # __slots__ = ("grid_size", "action")
    def __init__(
        self,
        init_position=None,
        min_mutation_amp=1,
        max_mutation_amp=5,
        grid_size=40,
        evolvability={
            "type":"perlin",
            "parameters":{
                "octaves":4,
                "seed":3
            }
        }):

        super().__init__(init_position)

        self.grid_size = grid_size
        
        self.min_mutation_amp = min_mutation_amp
        self.max_mutation_amp = max_mutation_amp

        self.evolvability = evolvability
    
    def __call__(self, *args, **kwds):
        super().__call__()

        return self.action
    
    def _make_amplitude(self, input_position=None):
        """
        Gives an amplitude to the agent's mutation

        Not very efficient to have it here rather than on the map.... fixed by using class's evo_map
        """

        if input_position is None:
            input_position = list(map(int,self.action[:]))

        if self.evolvability["type"] == "horizontal":
            amplitude = input_position[0]//10 + 1
        elif self.evolvability["type"] == "vertical":
            amplitude = input_position[1]//10 + 1
        elif self.evolvability["type"] == "diagonal":
            amplitude = sum(input_position) // 10 + 1
        elif self.evolvability["type"] == "band":
            spos = sum(input_position)
            amplitude = (spos // 5 + 1) if spos <= 20 else (4 - (spos-20) // 5)
        elif self.evolvability["type"] == "perlin":
            params = self.evolvability["parameters"]

            if GridAgentGuesser.evo_map is None:            
                noise = PerlinNoise(**params)
                GridAgentGuesser.evo_map = np.array([
                    [
                        noise((i/self.grid_size, j/self.grid_size))
                        for j in range(self.grid_size)
                    ] for i in range(self.grid_size)
                ])

                GridAgentGuesser.evo_map = np.tanh(self.evo_map/np.max(abs(self.evo_map)))

                GridAgentGuesser.evo_map = np.round((self.evo_map + np.tanh(.5)) * (self.max_mutation_amp - self.min_mutation_amp) + self.min_mutation_amp)
            
            amplitude = int(np.clip(self.evo_map[input_position[0], input_position[1]], self.min_mutation_amp, self.max_mutation_amp))
            
            assert amplitude >= self.min_mutation_amp and amplitude <= self.max_mutation_amp, "amplitude {} out of bounds".format(amplitude)

        else:
            random.seed(int(str(input_position[0]) + str(input_position[1])))
            amplitude = random.randint(self.min_mutation_amp, self.max_mutation_amp)

        return amplitude
    
    def mutate(self, mutation=None, amplitude=None):
        """
        A mutation is LEFT, RIGHT, UP, DOWN move
        """

        if mutation is None:
            mutation = random.choice(self.direction_list)
        
        # Seeding for evolvability
        if amplitude is None:
            amplitude = self._make_amplitude()
            
        if mutation == "LEFT":
            self.action = tuple([
                self.action[0],
                max(0, self.action[1] - amplitude)
            ])

        elif mutation == "RIGHT":
            self.action = tuple([
                self.action[0],
                min(self.grid_size - 1, self.action[1] + amplitude)
            ])

        elif mutation == "UP":
            self.action = tuple([
                max(0, self.action[0] - amplitude),
                self.action[1]
            ])

        elif mutation == "DOWN":
            self.action = tuple([
                min(self.grid_size - 1, self.action[0] + amplitude),
                self.action[1]
            ])
        
        self.behavior = self.action


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