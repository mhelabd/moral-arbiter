# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from moral_arbiter.foundation import utils
from moral_arbiter.foundation.agents import agent_registry as agents
from moral_arbiter.foundation.components import component_registry as components
from moral_arbiter.foundation.entities import endogenous_registry as endogenous
from moral_arbiter.foundation.entities import landmark_registry as landmarks
from moral_arbiter.foundation.entities import resource_registry as resources
from moral_arbiter.foundation.scenarios import scenario_registry as scenarios


def make_env_instance(scenario_name, **kwargs):
    scenario_class = scenarios.get(scenario_name)
    return scenario_class(**kwargs)
