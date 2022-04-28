import torch
import numpy as np


class GridAgent(torch.nn.Module):
    """
    A grid agent, able to take an action
    """

    id = 0

    def __init__(
        self,

        input_dim=2, output_dim=2,
        hidden_layers=3, hidden_dim=10,
        use_batchnorm=False,
        non_linearity=torch.tanh, 
        output_normalizer=lambda x: x,
        
        generation=0, parent=None,
        ):
        """
        Enter NN parameters aswell as agent's lineage
        """

        super().__init__()

        self.id = GridAgent.id
        GridAgent.id += 1

        # Agent's network
        self.mds = torch.nn.ModuleList([torch.nn.Linear(input_dim, hidden_dim)])

        for i in range(hidden_layers - 1):
            self.mds.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.mds.append(torch.nn.Linear(hidden_dim, output_dim))

        self.non_linearity = non_linearity
        self.batchnorm = torch.nn.BatchNorm1d(hidden_dim) if use_batchnorm else lambda x: x

        self.output_normaliser = output_normalizer

        # Family tree
        self.age = 0

        self.created_at_gen = generation
        self.parent = parent

        # Behavioral descriptor of the agent
        self.state_hist = None
        self.behavior = None

        self.weights = self.get_flattened_weights()
    
    def forward(self, x, return_type="tuple"):
        """
        x : list
        return type : tuple; numpy; torch
        """

        output = torch.Tensor(x).unsqueeze(0)
        output = self.non_linearity(self.mds[0](output))
        for md in self.mds[1:-1]:
            output = self.batchnorm(self.non_linearity(md(output)))

        output = self.non_linearity(self.mds[-1](output))
        
        output = self.output_normaliser(output)
        
        if return_type == "tuple":
            return tuple(*output.numpy())
        elif return_type == "numpy":
            return output.detach().cpu().numpy()
        else:
            return output

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
    
    def update_behavior(self, state_hist):
        """
        Updates the agent's behavior descriptor based on its trajectory
        """

        self.state_hist = state_hist
        self.behavior = self.state_hist[-1]
        return self.behavior

    def meta_update(self):
        """
        Updates the agent if it's selected for next meta-step
        """

        self.age += 1
    
    def __len__(self):
        return len(self.weights)
    
    def __getitem__(self, key):
        return self.weights[key]
    
    def __setitem__(self, key, value):
        new_weights = self.weights[:key] + [value] + self.weights[key+1:]
        self.set_flattened_weights(new_weights)


if __name__ == "__main__":

    from class_gridworld import GridWorld
    from utils_worlds import GridWorldSparse40x40Mixed
    from class_distribution_shapes import UniformRectangle
    
    GridWorldSparse40x40Mixed["start_distribution"] = UniformRectangle((0,0), 40, 40)
    g = GridWorld(**GridWorldSparse40x40Mixed, is_guessing_game=True)

    list_ag = [
        GridAgent(
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