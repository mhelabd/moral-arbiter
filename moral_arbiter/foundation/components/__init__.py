# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from moral_arbiter.foundation.base.base_component import component_registry

from . import (
    build,
    continuous_double_auction,
    move,
    redistribution,
    neural_morality,
    simple_labor,
    steal,
)

# Import files that add Component class(es) to component_registry
# ---------------------------------------------------------------
