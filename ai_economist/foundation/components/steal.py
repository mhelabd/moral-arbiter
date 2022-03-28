# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np
from numpy.random import rand

from ai_economist.foundation.base.base_component import (
    BaseComponent,
    component_registry,
)


@component_registry.add
class Steal(BaseComponent):
    """
    Allows mobile agents to steal resources and money from each other and prevents
    agents from stealing when no resources are available to steal.

    Can be configured to include stealling skill, where agents have heterogeneous
    probabilities of collecting bonus resources without additional labor cost.

    Args:
        steal_labor (float): Labor cost associated with movement. Must be >= 0.
            Default is 1.0.
        skill_dist (str): Distribution type for sampling skills. Default ("none")
            gives all agents identical skill equal to a bonus prob of 0. "pareto" and
            "lognormal" sample skills from the associated distributions.
    """

    name = "Steal"
    component_type = "Steal"
    required_entities = ["Coin", "House", "Labor"] # TODO: Add Other resources
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
        self,
        *base_component_args,
        steal_labor=1.0,
        skill_dist="none",
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        self.steal_labor = float(steal_labor)
        assert self.steal_labor >= 0

        self.skill_dist = skill_dist.lower()
        assert self.skill_dist in ["none", "pareto", "lognormal"]

        self.steals = []

    # Required methods for implementing components
    # --------------------------------------------

    def get_n_actions(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        Adds 1 action (steal) for mobile agents.
        """
        # This component adds 4 action that agents can take:
        # move up, down, left, or right
        if agent_cls_name == "BasicMobileAgent":
            return 1
        return None

    def get_additional_state_fields(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        For mobile agents, add state field for steal skill.
        """
        if agent_cls_name not in self.agent_subclasses:
            return {}
        if agent_cls_name == "BasicMobileAgent":
            return {"bonus_steal_prob": 0.0}
        raise NotImplementedError

    def component_step(self):
        """
        See base_component.py for detailed description.

        Move to adjacent, unoccupied locations. Collect resources when moving to
        populated resource tiles, adding the resource to the agent's inventory and
        de-populating it from the tile.
        """
        world = self.world

        steals = []
        for agent in world.get_random_order_agents():

            if self.name not in agent.action:
                return
            action = agent.get_component_action(self.name)

            r, c = [int(x) for x in agent.loc]

            if action == 0:  # NO-OP!
               pass
            elif action == 1:
                # Get surronding agent who is not you
                agents_near_r_c = world.agent_by_location(r, c, agent.idx)
                
                if agents_near_r_c == -1:
                    continue
                
                # Randomly choose someone to steal from
                agent_near_r_c = np.random.choice(agents_near_r_c)
                n_gathered = 1 + (rand() < agent.state["bonus_steal_prob"])
                # Check that you can steal from that agent
                n_gathered = min(n_gathered, agent_near_r_c.state["inventory"]["Coin"])
                agent_near_r_c.state["inventory"]["Coin"] -= n_gathered
                agent.state["inventory"]["Coin"] += n_gathered                
                agent.state["endogenous"]["Labor"] += self.steal_labor
                steals.append(
                    dict(
                        stealing_agent=agent.idx,
                        stolen_from_agent=agent_near_r_c.idx,
                        resource="coin",
                        income=n_gathered,
                        loc=[r, c],
                    )
                )
            else:
                raise ValueError
    
        self.steals.append(steals)


    def generate_observations(self):
        """
        See base_component.py for detailed description.

        Here, agents observe their steal skill. The planner does not observe
        anything from this component.
        """
        return {
            str(agent.idx): {"bonus_steal_prob": agent.state["bonus_steal_prob"]}
            for agent in self.world.agents
        }

    def generate_masks(self, completions=0):
        """
        See base_component.py for detailed description.

        Prevent stealing when no adjacent person is present
        """
        world = self.world

        masks = {}
        for agent in world.agents:
            r, c = agent.state["loc"]
            agents_near_r_c = world.agent_by_location(r, c, agent.idx)
            masks[agent.idx] = [0 if agents_near_r_c == -1 else 1]
        return masks

    # For non-required customization
    # ------------------------------

    def additional_reset_steps(self):
        """
        See base_component.py for detailed description.

        Re-sample agents' collection skills.
        """
        for agent in self.world.agents:
            if self.skill_dist == "none":
                bonus_rate = 0.0
            elif self.skill_dist == "pareto":
                bonus_rate = np.minimum(2, np.random.pareto(3)) / 2
            elif self.skill_dist == "lognormal":
                bonus_rate = np.minimum(2, np.random.lognormal(-2.022, 0.938)) / 2
            else:
                raise NotImplementedError
            agent.state["bonus_steal_prob"] = float(bonus_rate)

        self.steals = []

    def get_dense_log(self):
        """
        Log resource collections.

        Returns:
            gathers (list): A list of gather events. Each entry corresponds to a single
                timestep and contains a description of any resource gathers that
                occurred on that timestep.

        """
        return self.steals
