""" 
Definition of the HyperParameters class, which contains all the hyperparameters of the model.
It also contains hyperparameters useful for constructing the graph, such as the linking radius.

"""

""" 
Definition of the HyperParameters class, which contains all the hyperparameters of the model.
It also contains hyperparameters useful for constructing the graph, such as the linking radius.
"""

class HyperParameters:
    def __init__(   self,
                    simsuite,           # Simulation suite, choose between "IllustrisTNG" and "SIMBA"
                    targetsuite,
                    simset,             # Simulation set, choose between "CV" and "LH"
                    n_sims,             # Number of simulations considered, maximum 27 for CV and 1000 for LH 
                    domain_adapt,       # Domain Adaptation type,
                    da_cond_loss_fraction,   # Fraction of the da loss that is the conditional da loss
                    training,           # If training, set to True, otherwise loads a pretrained model and tests it
                    pred_params,        # Number of cosmo/astro params to be predicted, starting from Omega_m, sigma_8, etc.
                    only_positions,     # 1 for using only positions as features, 0 for using additional galactic features
                    snap,               # Snapshot of the simulation, indicating redshift 4: z=3, 10: z=2, 14: z=1.5, 18: z=1, 24: z=0.5, 33: z=0
                    da_loss_fraction,   # Fraction of the total loss that is the MMD loss
                    r_link,             # Linking radius to build the graph
                    n_layers,           # Number of graph layers
                    hidden_channels,    # Hidden channels
                    n_epochs,           # Number of epochs
                    learning_rate,      # Learning rate
                    weight_decay,       # Weight decay
                    weight_da,          # Domain adaptation weight
                    seed,
                    model_select        # choose model type 
                ):
        # Set non optimizable hyperparameters (construction choices)
        self.simsuite = simsuite
        self.targetsuite = targetsuite
        self.simset = simset
        self.n_sims = n_sims
        self.domain_adapt = domain_adapt
        self.da_cond_loss_fraction = da_cond_loss_fraction
        self.training = training
        self.pred_params = pred_params
        self.only_positions = only_positions
        self.snap = snap
        self.da_loss_fraction = da_loss_fraction
        self.seed = seed
        self.model_select = model_select

        # Set optimizable hyperparameters
        self.r_link = r_link
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.weight_da = weight_da

    def __str__(self):
        """Return a formatted string of all hyperparameters."""
        # Option 1: Using f-strings
        return (
            f"Hyperparameters:\n"
            f"Simulation suite: {self.simsuite}\n"
            f"Target suite: {self.targetsuite}\n"
            f"Simulation set: {self.simset}\n"
            f"Number of simulations: {self.n_sims}\n"
            f"Domain adaptation: {self.domain_adapt}\n"
            f"DA conditional loss fraction: {self.da_cond_loss_fraction}\n"
            f"Training: {self.training}\n"
            f"Number of predicted parameters: {self.pred_params}\n"
            f"Only positions: {self.only_positions}\n"
            f"Snapshot: {self.snap}\n"
            f"DA loss fraction: {self.da_loss_fraction}\n"
            f"Linking radius: {self.r_link}\n"
            f"Number of graph layers: {self.n_layers}\n"
            f"Hidden channels: {self.hidden_channels}\n"
            f"Number of epochs: {self.n_epochs}\n"
            f"Learning rate: {self.learning_rate}\n"
            f"Weight decay: {self.weight_decay}\n"
            f"Domain adaptation weight: {self.weight_da}\n"
            f"Seed: {self.seed}\n"
            f"Model Selection: {self.model_select}\n"
        )

        # Option 2: Dynamic Attribute Iteration
        # params = "\n".join([f"{key}: {value}" for key, value in self.__dict__.items()])
        # return f"Hyperparameters:\n{params}"

    def name_model(self):
        """Generate the name of the model for file saving.

        Returns:
            str: Name of the model.
        """
        # Fixing the simsuite join issue. Since simsuite is a string, no need to join its characters.
        # If simsuite can be a list in some cases, handle accordingly.
        if isinstance(self.simsuite, list):
            simsuite_str = f"[{', '.join(self.simsuite)}]"
        else:
            simsuite_str = self.simsuite

        return (
            f"{self.model_select}_"
            f"{simsuite_str}_"
            f"{self.targetsuite}_"
            f"{self.simset}_"
            f"{self.domain_adapt}_FCR_{self.da_cond_loss_fraction}_"
            f"FR_{self.da_loss_fraction}_onlypos_{self.only_positions}_"
            f"lr_{self.learning_rate:.2e}_weight-da_{self.weight_da:.2e}_"
            f"weightdecay_{self.weight_decay:.2e}_layers_{self.n_layers}_"
            f"rlink_{self.r_link:.2e}_channels_{self.hidden_channels}_"
            f"epochs_{self.n_epochs}"
        )

    def flip_suite(self):
        """Return the other CAMELS simulation suite.

        Returns:
            str: Other CAMELS simulation suite.
        """
        return self.targetsuite 