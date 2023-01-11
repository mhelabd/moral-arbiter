# Using Multi-Agent Reinforcement Learning to Simulate Moral Theories

This repo contains an implementation of a complex multi-agent economic system as well as codifications for three moral theories: virtue ethics, utilitarianism, and the moral arbiter. 

### Installing from Source

1. Clone this repository to your local machine:

  ```
   git clone www.github.com/mhelabd/moral-arbiter
   ```

2. Create a new conda environment (named "moral-arbiter" below - replace with anything else) and activate it

  ```pyfunctiontypecomment
   conda create --name moral-arbiter python=3.7 --yes
   conda activate moral-arbiter
   ```

3. Either

   a) Edit the PYTHONPATH to include the moral-arbiter directory
  ```
   export PYTHONPATH=<local path to moral-arbiter>:$PYTHONPATH
   ```

   OR

   b) Install as an editable Python package
  ```pyfunctiontypecomment
   cd moral-arbiter
   pip install -e .
   ```

Useful tip: for quick access, add the following to your ~/.bashrc or ~/.bash_profile:

```pyfunctiontypecomment
alias moralarbiter ="conda activate moral-arbiter; cd <local path to moral-arbiter>"
```

You can then simply run `moralarbiter` once to activate the conda environment.

### Testing your Install

To test your installation, try running:

```
conda activate moral-arbiter
python -c "import moral_arbiter"
```

## Getting Started

## Structure of the Code

- The simulation is located in the `moral_arbiter/foundation` folder.

The code repository is organized into the following components:

| Component | Description |
| --- | --- |
| [base](https://www.github.com/mhelabd/moral-arbiter/blob/master/moral_arbiter/foundation/base) | Contains base classes to can be extended to define Agents, Components and Scenarios. |
| [agents](https://www.github.com/mhelabd/moral-arbiter/blob/master/moral_arbiter/foundation/agents) | Agents represent economic actors in the environment. Currently, we have mobile Agents (representing workers) and a social planner (representing a government). |
| [entities](https://www.github.com/mhelabd/moral-arbiter/blob/master/moral_arbiter/foundation/entities) | Endogenous and exogenous components of the environment. Endogenous entities include labor, while exogenous entity includes landmarks (such as Water and Grass) and collectible Resources (such as Wood and Stone). |
| [components](https://www.github.com/mhelabd/moral-arbiter/blob/master/moral_arbiter/foundation/components) | Components are used to add some particular dynamics to an environment. They also add action spaces that define how Agents can interact with the environment via the Component. |
| [scenarios](https://www.github.com/mhelabd/moral-arbiter/blob/master/moral_arbiter/foundation/scenarios) | Scenarios compose Components to define the dynamics of the world. It also computes rewards and exposes states for visualization. |
