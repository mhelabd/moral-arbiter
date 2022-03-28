# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from ai_economist.foundation.scenarios.utils import social_metrics

def isoelastic_coin_minus_labor(
    coin_endowment, total_labor, isoelastic_eta, labor_coefficient
):
    """Agent utility, concave increasing in coin and linearly decreasing in labor.

    Args:
        coin_endowment (float, ndarray): The amount of coin owned by the agent(s).
        total_labor (float, ndarray): The amount of labor performed by the agent(s).
        isoelastic_eta (float): Constant describing the shape of the utility profile
            with respect to coin endowment. Must be between 0 and 1. 0 yields utility
            that increases linearly with coin. 1 yields utility that increases with
            log(coin). Utility from coin uses:
                https://en.wikipedia.org/wiki/Isoelastic_utility
        labor_coefficient (float): Constant describing the disutility experienced per
            unit of labor performed. Disutility from labor equals:
                labor_coefficient * total_labor

    Returns:
        Agent utility (float) or utilities (ndarray).
    """
    # https://en.wikipedia.org/wiki/Isoelastic_utility
    assert np.all(coin_endowment >= 0)
    assert 0 <= isoelastic_eta <= 1.0

    # Utility from coin endowment
    if isoelastic_eta == 1.0:  # dangerous
        util_c = np.log(np.max(1, coin_endowment))
    else:  # isoelastic_eta >= 0
        util_c = (coin_endowment ** (1 - isoelastic_eta) - 1) / (1 - isoelastic_eta)

    # disutility from labor
    util_l = total_labor * labor_coefficient

    # Net utility
    util = util_c - util_l

    return util

def coin_minus_labor_cost(
    coin_endowment, total_labor, labor_exponent, labor_coefficient
):
    """Agent utility, linearly increasing in coin and decreasing as a power of labor.

    Args:
        coin_endowment (float, ndarray): The amount of coin owned by the agent(s).
        total_labor (float, ndarray): The amount of labor performed by the agent(s).
        labor_exponent (float): Constant describing the shape of the utility profile
            with respect to total labor. Must be between >1.
        labor_coefficient (float): Constant describing the disutility experienced per
            unit of labor performed. Disutility from labor equals:
                labor_coefficient * total_labor.

    Returns:
        Agent utility (float) or utilities (ndarray).
    """
    # https://en.wikipedia.org/wiki/Isoelastic_utility
    assert np.all(coin_endowment >= 0)
    assert labor_exponent > 1

    # Utility from coin endowment
    util_c = coin_endowment

    # Disutility from labor
    util_l = (total_labor ** labor_exponent) * labor_coefficient

    # Net utility
    util = util_c - util_l

    return util

def coin_eq_times_productivity(coin_endowments, equality_weight):
    """Social welfare, measured as productivity scaled by the degree of coin equality.

    Args:
        coin_endowments (ndarray): The array of coin endowments for each of the
            agents in the simulated economy.
        equality_weight (float): Constant that determines how productivity is scaled
            by coin equality. Must be between 0 (SW = prod) and 1 (SW = prod * eq).

    Returns:
        Product of coin equality and productivity (float).
    """
    n_agents = len(coin_endowments)
    prod = social_metrics.get_productivity(coin_endowments) / n_agents
    equality = equality_weight * social_metrics.get_equality(coin_endowments) + (
        1 - equality_weight
    )
    return equality * prod

def inv_income_weighted_coin_endowments(coin_endowments):
    """Social welfare, as weighted average endowment (weighted by inverse endowment).

    Args:
        coin_endowments (ndarray): The array of coin endowments for each of the
            agents in the simulated economy.

    Returns:
        Weighted average coin endowment (float).
    """
    pareto_weights = 1 / np.maximum(coin_endowments, 1)
    pareto_weights = pareto_weights / np.sum(pareto_weights)
    return np.sum(coin_endowments * pareto_weights)

def inv_income_weighted_utility(coin_endowments, utilities):
    """Social welfare, as weighted average utility (weighted by inverse endowment).

    Args:
        coin_endowments (ndarray): The array of coin endowments for each of the
            agents in the simulated economy.
        utilities (ndarray): The array of utilities for each of the agents in the
            simulated economy.

    Returns:
        Weighted average utility (float).
    """
    pareto_weights = 1 / np.maximum(coin_endowments, 1)
    pareto_weights = pareto_weights / np.sum(pareto_weights)
    return np.sum(utilities * pareto_weights)

def utilitarian_coin_minus_labor_cost(
    coin_endowment, total_labor, labor_exponent, labor_coefficient, coin_endowments, utilitarian_coefficient, 
):
    """Agent utility, linearly increasing in coin and decreasing as a power of labor.

    Args:
        coin_endowment (float, ndarray): The amount of coin owned by the agent(s).
        total_labor (float, ndarray): The amount of labor performed by the agent(s).
        labor_coefficient (float): Constant describing the disutility experienced per
            unit of labor performed. Disutility from labor equals:
                labor_coefficient * total_labor.
        coin_endowments (ndarray): The array of coin endowments for each of the
            agents in the simulated economy.
        utilitarian_coefficient (float): Constant describing the utility experienced per
            unit of utility added to other members. Utility from utilitarianism:
                utilitarian_coefficient * total_utility.

    Returns:
        Agent utility (float) or utilities (ndarray).
    """

    assert np.all(coin_endowment >= 0)
    assert labor_exponent > 1
    total_utility = social_metrics.total_utility(coin_endowments)
    num_agents = len(coin_endowments)
    util_other_agents = (utilitarian_coefficient/num_agents) * total_utility

    # Utility from coin endowment
    util_c = coin_endowment

    # Disutility from labor
    util_l = (total_labor ** labor_exponent) * labor_coefficient

    # Net utility
    util = util_c - util_l + util_other_agents

    return util

def virtue_ethics_coin_minus_labor_cost(
    coin_endowment, total_labor, labor_exponent, labor_coefficient, is_moral_action, virtue_ethics_coefficient, 
):
    """Agent utility, linearly increasing in coin and decreasing as a power of labor.

    Args:
        coin_endowment (float, ndarray): The amount of coin owned by the agent(s).
        total_labor (float, ndarray): The amount of labor performed by the agent(s).
        labor_coefficient (float): Constant describing the disutility experienced per
            unit of labor performed. Disutility from labor equals:
                labor_coefficient * total_labor.
        coin_endowments (ndarray): The array of coin endowments for each of the
            agents in the simulated economy.
        virtue_ethics_coefficient (float): Constant describing the reward/penality experienced per
            action given its morality. 

    Returns:
        Agent utility (float) or utilities (ndarray).
    """

    assert np.all(coin_endowment >= 0)
    assert labor_exponent > 1
    is_moral_action = 1 if is_moral_action == 1 else -1
    virtue_ethics_reward = is_moral_action * virtue_ethics_coefficient         

    # Utility from coin endowment
    util_c = coin_endowment

    # Disutility from labor
    util_l = (total_labor ** labor_exponent) * labor_coefficient

    # Net utility
    util = util_c - util_l + virtue_ethics_reward

    return util

def utilitarian_isoelastic_coin_minus_labor(
    coin_endowment, total_labor, isoelastic_eta, labor_coefficient, coin_endowments, utilitarian_coefficient, 
):
    """Agent utility, concave increasing in coin, linearly decreasing in labor 
    and linearly increasing in total utility.

    Args:
        coin_endowment (float, ndarray): The amount of coin owned by the agent(s).
        total_labor (float, ndarray): The amount of labor performed by the agent(s).
        isoelastic_eta (float): Constant describing the shape of the utility profile
            with respect to coin endowment. Must be between 0 and 1. 0 yields utility
            that increases linearly with coin. 1 yields utility that increases with
            log(coin). Utility from coin uses:
                https://en.wikipedia.org/wiki/Isoelastic_utility
        labor_coefficient (float): Constant describing the disutility experienced per
            unit of labor performed. Disutility from labor equals:
                labor_coefficient * total_labor
        coin_endowments (ndarray): The array of coin endowments for each of the
            agents in the simulated economy.
        utilitarian_coefficient (float): Constant describing the utility experienced per
            unit of utility added to other members. Utility from utilitarianism:
                utilitarian_coefficient * total_utility.

    Returns:
        Agent utility (float) or utilities (ndarray).
    """
    # https://en.wikipedia.org/wiki/Isoelastic_utility
    assert np.all(coin_endowment >= 0)
    assert 0 <= isoelastic_eta <= 1.0
    total_utility = social_metrics.total_utility(coin_endowments)
    num_agents = len(coin_endowments)
    util_other_agents = (utilitarian_coefficient/num_agents) * total_utility
    util_c = coin_endowment + util_other_agents

    # Utility from coin endowment
    if isoelastic_eta == 1.0:  # dangerous
        util_c = np.log(np.max(1, util_c))
    else:  # isoelastic_eta >= 0
        util_c = (util_c ** (1 - isoelastic_eta) - 1) / (1 - isoelastic_eta)

    # disutility from labor
    util_l = total_labor * labor_coefficient

    # Net utility
    util = util_c - util_l

    return util

def virtue_ethics_isoelastic_coin_minus_labor(
    coin_endowment, total_labor, isoelastic_eta, labor_coefficient, 
    is_moral_action, virtue_ethics_coefficient, 
):
    """Agent utility, concave increasing in coin, linearly decreasing 
    with constant virtue ethics reward.

    Args:
        coin_endowment (float, ndarray): The amount of coin owned by the agent(s).
        total_labor (float, ndarray): The amount of labor performed by the agent(s).
        isoelastic_eta (float): Constant describing the shape of the utility profile
            with respect to coin endowment. Must be between 0 and 1. 0 yields utility
            that increases linearly with coin. 1 yields utility that increases with
            log(coin). Utility from coin uses:
                https://en.wikipedia.org/wiki/Isoelastic_utility
        labor_coefficient (float): Constant describing the disutility experienced per
            unit of labor performed. Disutility from labor equals:
                labor_coefficient * total_labor
        coin_endowments (ndarray): The array of coin endowments for each of the
            agents in the simulated economy.
        utilitarian_coefficient (float): Constant describing the utility experienced per
            unit of utility added to other members. Utility from utilitarianism:
                utilitarian_coefficient * total_utility.

    Returns:
        Agent utility (float) or utilities (ndarray).
    """
    # https://en.wikipedia.org/wiki/Isoelastic_utility
    assert np.all(coin_endowment >= 0)
    assert 0 <= isoelastic_eta <= 1.0
    is_moral_action = 1 if is_moral_action == 1 else -1
    virtue_ethics_reward = is_moral_action * virtue_ethics_coefficient         
    util_c = coin_endowment

    # Utility from coin endowment
    if isoelastic_eta == 1.0:  # dangerous
        util_c = np.log(np.max(1, util_c))
    else:  # isoelastic_eta >= 0
        util_c = (util_c ** (1 - isoelastic_eta) - 1) / (1 - isoelastic_eta)

    # disutility from labor
    util_l = total_labor * labor_coefficient  

    # Net utility
    util = util_c - util_l + virtue_ethics_reward

    return util

def learned_reward(planner, coin_endowments, labors, stone_endowments, wood_endowments, labor_coefficient, isoelastic_eta, moral_coeffecient):
    """Agent utility, concave increasing in learned utility, linearly decreasing 
    with labor.

    Args:
        coin_endowment (float, ndarray): The amount of coin owned by the agent(s).
        labors (float, ndarray): The amount of labor done by the agent(s).
        stone_endowments (float, ndarray): The amount of stones earned by the agent(s).
        wood_endowments (float, ndarray): The amount of woods earned by the agent(s).
        labor_coefficient (float): Constant describing the disutility experienced per
            unit of labor performed. Disutility from labor equals:
                labor_coefficient * total_labor
        isoelastic_eta (float): Constant describing the shape of the utility profile
            with respect to coin endowment. Must be between 0 and 1. 0 yields utility
            that increases linearly with coin. 1 yields utility that increases with
            log(coin). Utility from coin uses:
                https://en.wikipedia.org/wiki/Isoelastic_utility
        moral_coefficient (float): Constant describing how much an agent cares about themselves versus others.

    Returns:
        Agent utility (float) 
    """
    # https://en.wikipedia.org/wiki/Isoelastic_utility
    coin_endowment = coin_endowments[0]
    total_labor = labors[0]
    # world.planner.get_state_multiplier
    assert np.all(coin_endowment >= 0)
    assert 0 <= isoelastic_eta <= 1.0
    learned_utility = social_metrics.learned_utility(planner, coin_endowments, labors, stone_endowments, wood_endowments)
    # how much to value my own coin endowment versus the learned utility
    util_c = (1 -  moral_coeffecient) * coin_endowment + moral_coeffecient * learned_utility

    # Utility from coin endowment
    if isoelastic_eta == 1.0:  # dangerous
        util_c = np.log(np.max(1, util_c))
    else:  # isoelastic_eta >= 0
        util_c = (util_c ** (1 - isoelastic_eta) - 1) / (1 - isoelastic_eta)

    # disutility from labor
    util_l = total_labor * labor_coefficient

    # Net utility
    util = util_c - util_l

    return util
