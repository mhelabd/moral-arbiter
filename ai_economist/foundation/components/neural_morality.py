from copy import deepcopy

import numpy as np

from ai_economist.foundation.base.base_component import (
    BaseComponent,
    component_registry,
)


@component_registry.add
class NeuralMorality(BaseComponent):
    """Defines agent Morality

    Note:
        If this component is used, it should always be the last component in the order!

    Args:
        disable_morality (bool): Whether to disable any enforced morality, effectively
            enforcing every agent to focus on their own interest. Useful for removing morality without
            changing the observation space. Default is False (Morality enabled).
        period (int): Length of time between updating morality.
         Must be > 0. Default is 100 timesteps.
        bottom_moral_value (float): Value of the worst moral action
        top_moral_value (float): Value of the best moral action
        n_buckets (int): Number of spaces between bottom and top moral values
        states (list): List of states that the planner takes into account
        bucket_spacing (str): How bucket cutoffs should be spaced.
            "linear" linearly spaces the n_buckets cutoffs between bottom_moral_value and
                top_moral_value;
            "log" is similar to "linear" but with logarithmic spacing.
    """

    name = "NeuralMorality"
    component_type = "NeuralMorality"
    required_entities = ["Wood", "Stone", "Coin", "build_skill", "bonus_steal_prob", "Labor"]
    agent_subclasses = ["BasicMobileAgent", "BasicPlanner"]

    def __init__(
        self,
        *base_component_args,
        disable_morality=False,
        period=10,
        bottom_moral_value=-10,
        top_moral_value=10,
        n_buckets=10,
        bucket_spacing="log",
        states=['Coin', 'Wood', 'Stone', 'Labor'], #TODO add looking at the map
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        # Whether to turn off taxes. Disabling taxes will prevent any taxes from
        # being collected but the observation space will be the same as if taxes were
        # enabled, which can be useful for controlled tax/no-tax comparisons.
        self.disable_morality = bool(disable_morality)
        self.bottom_moral_value = bottom_moral_value
        self.top_moral_value = top_moral_value
        self.n_buckets = n_buckets
        #coins1, coins2, etc
        self.state_types = states
        # IMP Every agent sees themselves as the first agent, and then agents are ordered in order of skill
        self.states = [state + str(i) for i in range(self.world.n_agents) for state in states]

        self.curr_cycle_pos = 1

        # Number of states the planner is using for calculating rewards
        self.n_states = len(self.states) 
        self.bucket_spacing = bucket_spacing

        if self.bucket_spacing == "linear":
            self.bucket_cutoffs = np.linspace(
                self.bottom_moral_value, self.top_moral_value, self.n_buckets
            )
        elif self.bucket_spacing == "log":
            self.bucket_cutoffs = np.logspace(
              self.bottom_moral_value, self.top_moral_value, num=self.n_buckets
            )
        else:
          raise NotImplementedError(
            "Can Not handle {}".format(self.bucket_spacing)
          )

        # How many timesteps a moral period lasts.
        self.period = int(period)
        assert self.period > 0

        self.curr_bracket_tax_rates = np.zeros_like(self.bucket_cutoffs)
        self.curr_moral_values = [0 for _ in range(self.n_states)]

        self._planner_masks = None
        self.morals = []


    # Methods for getting/setting moral values
    # ----------------------------------------------

    def set_new_period_moral_code(self):
        """Update moral code using actions from the moral model."""
        if self.disable_morality:
            return

        for i, state in enumerate(self.states):
          planner_action = self.world.planner.get_component_action(
            self.name, "moralityof_{}".format(state)
          )
          if planner_action == 0:
            pass
          elif planner_action <= self.n_buckets:
            self.curr_moral_values[i] = int(planner_action - 1) # since 0 is taken
          else:
            raise ValueError

          self.morals.append({state: self.curr_moral_values[i] for i, state in enumerate(self.states)})


    # Required methods for implementing components
    # --------------------------------------------

    def get_n_actions(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        The planner's action space includes an action subspace for each of the states. Each
        such action space has as many actions as there are discretized moral values.
        """
        # Only the planner takes actions through this component.
        if agent_cls_name == "BasicPlanner" and not self.disable_morality:
                # For every state, the planner can select one of the discretized
                # moral rates.
                return [
                    ("moralityof_{:03d}".format(state), self.n_buckets)
                    for state in self.states
                ]

        # Return 0 (no added actions) if the other conditions aren't met.
        return 0

    def get_additional_state_fields(self, agent_cls_name):
        """This component does not add any agent state fields."""
        return {}

    def component_step(self):
        """
        See base_component.py for detailed description.

        On the first day of each moral period, update rewards. On the last day, restart it.
        """

        # 1. On the first day of a new moral rate: Set up the rewards for this period.
        if self.curr_cycle_pos == 1:
          self.set_new_period_rates_model()

        # 2. On the last day of the moral period, return to first position.
        if self.curr_cycle_pos >= self.period:
            self.curr_cycle_pos = 0

        # increment timestep.
        self.curr_cycle_pos += 1

    def generate_observations(self):
        """
        See base_component.py for detailed description.

        Agents observe where in the cycle they are, information about the 
        current moral values is also presented.

        The planner observes the same type of information, but for all the agents. It
        also sees, for each agent, their marginal tax rate and reported income from
        the previous tax period.
        """
        is_first_day = float(self.curr_cycle_pos == 1)
        phase = self.curr_cycle_pos / self.period

        obs = dict()

        obs[self.world.planner.idx] = dict(
            is_first_day=is_first_day,
            phase=phase,
            curr_moral_values=self.curr_moral_values,
        )

        for agent in self.world.agents:
            i = agent.idx
            k = str(i)
            agent_states = {}
            for state in self.state_types:
              if state == "Coin" or state == "Wood" or state == "Stone":
                agent_states[state] = agent.state["inventory"][state]
              elif state == "Labor":
                agent_states[state] = agent.state["endogenous"][state]
              else:
                agent_states[state] = agent.state[state]

            obs[k] = dict(
              is_first_day=is_first_day,
              tax_phase=phase,
              curr_rates=self.curr_moral_values,
            )

            obs["p" + k] = agent_states

        return obs

    def generate_masks(self, completions=0):
        """
        See base_component.py for detailed description.

        Masks only apply to the planner and if tax_model == "model_wrapper" and taxes
        are enabled.
        All tax actions are masked (so, only NO-OPs can be sampled) on all timesteps
        except when self.tax_cycle_pos==1 (meaning a new tax period is starting).
        When self.tax_cycle_pos==1, tax actions are masked in order to enforce any
        tax annealing.
        """
        if self.disable_morality:
          return {}

        if self._planner_masks is None:
            masks = super().generate_masks(completions=completions)
            self._planner_masks = dict(
                new_morals=deepcopy(masks[self.world.planner.idx]),
                zeros={
                    k: np.zeros_like(v)
                    for k, v in masks[self.world.planner.idx].items()
                },
            )

        # No need to recompute. Use the cached masks.
        masks = dict()
        if self.curr_cycle_pos != 1 or self.disable_morality:
            # Apply zero masks for any timestep where taxes
            # are not going to be updated.
            masks[self.world.planner.idx] = self._planner_masks["zeros"]
        else:
            masks[self.world.planner.idx] = self._planner_masks["new_morals"]

        return masks

    # For non-required customization
    # ------------------------------

    def additional_reset_steps(self):
        """
        See base_component.py for detailed description.

        Reset trackers.
        """
        self.curr_moral_values = [0 for _ in range(self.n_states)]

        self.curr_cycle_pos = 1
        self.morals = []

    def get_metrics(self):
        """
        See base_component.py for detailed description.

        Return metrics related to codifying a moral code.
        """
        out = dict()


        if not self.disable_morality:
            out = self.morals[-1]
        return out

    def get_dense_log(self):
        """
        Log taxes.

        Returns:
            taxes (list): A list of tax collections. Each entry corresponds to a single
                timestep. Entries are empty except for timesteps where a tax period
                ended and taxes were collected. For those timesteps, each entry
                contains the tax schedule, each agent's reported income, tax paid,
                and redistribution received.
                Returns None if taxes are disabled.
        """
        if self.disable_morality:
            return None
        return self.morals
