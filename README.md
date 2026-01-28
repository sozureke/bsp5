# AEGIS: Multi-Agent RL for Communication-Deduction Games

A production-quality multi-agent reinforcement learning project implementing a communication-deduction game with trust-mediated communication. Features PettingZoo + Gymnasium environments, NetworkX-based pathfinding, and CleanRL PPO training.

## Features

- **Two Roles**: Survivors and Impostors with distinct objectives and actions
- **Trust-Based Communication**: Discrete actions (SUPPORT, ACCUSE, DEFEND, QUESTION) that influence voting through a trust matrix
- **Partial Observability**: Configurable vision radius limits what agents can see
- **Meeting/Voting**: Body reports trigger discussion and voting phases with confidence-based filtering

## Installation

```bash
# Install in development mode
cd aegis
pip install -e .
```

## Quick Start

### Train a Single Experiment

```bash
# Training with config file
python -m aegis.train.rl_ppo -c aegis/train/configs/e3_trust_comm.config.yaml
```

### Run Multiple Experiments

```bash
# Run experiment with all seeds (200 seeds)
python -m aegis.scripts.run_experiments e3_trust_comm.config.yaml

# Run multiple experiments
python -m aegis.scripts.run_experiments e1_baseline.config.yaml e3_trust_comm.config.yaml

# Run with test seeds (first 10 seeds, for quick testing)
python -m aegis.scripts.run_experiments e1_baseline.config.yaml --test
```

## Experiments

The project includes controlled ablation experiments studying trust-mediated communication:

| Experiment | Config File | Description | Key Variable |
|------------|-------------|-------------|--------------|
| **E1: Baseline** | `e1_baseline.config.yaml` | No communication, no trust | `enable_trust_comm: false` |
| **E2: FoV Low** | `e2_fov_low.config.yaml` | Limited visibility with communication | `vision_radius: 1` |
| **E2: FoV High** | `e2_fov_high.config.yaml` | Extended visibility with communication | `vision_radius: 3` |
| **E3: Trust + Comm** | `e3_trust_comm.config.yaml` | Main condition with trust filtering | `trust_delay_ticks: 1` |
| **E4: Delay Immediate** | `e4_delay_immediate.config.yaml` | Immediate trust updates | `trust_delay_ticks: 0` |
| **E4: Delay Delayed** | `e4_delay_delayed.config.yaml` | Delayed trust updates | `trust_delay_ticks: 5` |

Additional experiments (E5, E6, E7) are available in `train/configs/extra/`.

## Analysis

After running experiments, analyze results:

```bash
# Merge events from all seeds
python -m aegis.analysis.merge_events checkpoints/e3_trust_comm

# Aggregate metrics from all experiments
python -m aegis.analysis.aggregate_all_metrics

# Generate plots
python -m aegis.analysis.plot_metrics

# Analyze events
python -m aegis.analysis.analyze_events checkpoints/e3_trust_comm/events_all.jsonl --summary
```

## Project Structure

```
aegis/
├── core/              # Game engine, world state, rules
├── obs/               # Observation builders and spaces
├── comms/             # Communication actions and trust system
├── agents/            # Agent policies (heuristic, random)
├── logging/           # Event logging and metrics
├── train/             # PPO training and experiment configs
│   ├── rl_ppo.py      # Main training entry point
│   ├── seeds.py       # Shared seed set for experiments
│   └── configs/       # Experiment configurations
│       └── extra/     # Additional experiments (E5, E6, E7)
├── scripts/           # Experiment execution
│   └── run_experiments.py
└── analysis/          # Result analysis and visualization
    ├── aggregate_all_metrics.py
    ├── analyze_events.py
    ├── merge_events.py
    ├── plot_metrics.py
    └── evaluate_ppo.py
```

## Results

Results are saved in:
- `checkpoints/<experiment>/seed_<N>/` - Individual seed results
- `checkpoints/<experiment>/events_all.jsonl` - Merged events (after running merge_events)
- `results/all_metrics.json` - Aggregated metrics (after running aggregate_all_metrics)
- `results/plots/` - Visualization plots (after running plot_metrics)

## Fixed Parameters

All experiments use identical parameters except for the variables being tested:
- **Agents**: 4 survivors + 2 impostors per episode
- **Map**: 9 rooms (3×3 grid), evac_room=4
- **Episode length**: Maximum 600 ticks
- **Meetings**: 80 ticks discussion + 40 ticks voting
- **Training**: 2,000,000 timesteps, PPO with fixed hyperparameters
- **Seeds**: Shared set of 200 seeds for paired comparison
- **Vision radius**: Fixed to 2 (except E2: 1 and 3)

## License

MIT License

## Author

Marshanishvili Mykhailo, University of Luxembourg
