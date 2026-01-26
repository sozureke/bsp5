# AEGIS: Multi-Agent RL for Communication-Deduction Games

A production-quality multi-agent reinforcement learning project implementing a communication-deduction game. Features PettingZoo + Gymnasium environments, NetworkX-based pathfinding, and CleanRL PPO training.

## Features

- **Two Roles**: Survivors and Impostors with distinct objectives and actions
- **Communication System**: Discrete token-based messaging during meetings
- **Trust-Based Communication**: Optional discrete actions (SUPPORT, ACCUSE, DEFEND, QUESTION) that influence voting through a trust matrix
- **Task System**: Single-point and two-step tasks for survivors
- **Meeting/Voting**: Body reports trigger discussion and voting phases
- **Partial Observability**: Configurable vision radius limits what agents can see
- **Door Sabotage**: Impostors can temporarily close doors
- **Knower Modifier**: Optional role that knows one other agent's identity

## Installation

```bash
# Clone the repository
cd aegis

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install gymnasium pettingzoo supersuit networkx torch numpy pyyaml
```

## Quick Start

### Train with CleanRL PPO

```bash
# Training with config file
python -m aegis.train.rl_ppo -c aegis/train/configs/main.config.yaml
```

## Experiments

The project includes a comprehensive set of experiments designed to study trust and communication in multi-agent hidden role games. Each experiment answers a specific research question while keeping all other parameters constant.

| Experiment | Config File | Description | Key Variable | Research Question |
|------------|-------------|-------------|--------------|------------------|
| **E1: Baseline** | `e1_baseline.config.yaml` | No communication, no trust. Agents vote based solely on event-based suspicion. | `enable_trust_comm: false`, `trust_voting_weight: 0.0` | What is the baseline performance without social mechanisms? |
| **E2: Field of View (Low)** | `e2_fov_low.config.yaml` | Limited visibility (vision_radius: 1) with full communication system. | `vision_radius: 1`, `enable_trust_comm: true` | How does communication compensate for limited sensory information? |
| **E2: Field of View (High)** | `e2_fov_high.config.yaml` | Extended visibility (vision_radius: 3) with full communication system. | `vision_radius: 3`, `enable_trust_comm: true` | How does increased observability affect communication necessity? |
| **E3: Trust + Communication** | `e3_trust_comm.config.yaml` | Full system: discrete communication (SUPPORT/ACCUSE/DEFEND/QUESTION) with delayed trust updates. | `enable_trust_comm: true`, `trust_delay_ticks: 1` | Does discrete communication improve hidden role detection? |
| **E4: Delay (Immediate)** | `e4_delay_immediate.config.yaml` | Trust updates apply immediately (no delay). | `trust_delay_ticks: 0` | Does immediate trust enable manipulation? |
| **E4: Delay (Delayed)** | `e4_delay_delayed.config.yaml` | Trust updates apply after delay. | `trust_delay_ticks: 5` | Does delay prevent voting manipulation? |
| **E5: Trust Weight (0.0)** | `e5_trust_weight_0.config.yaml` | Pure suspicion-based voting (no trust influence). | `trust_voting_weight: 0.0` | What happens when trust has no voting influence? |
| **E5: Trust Weight (Low)** | `e5_trust_weight_low.config.yaml` | Low trust influence in voting. | `trust_voting_weight: 0.3` | What is the optimal trust weight? |
| **E5: Trust Weight (High)** | `e5_trust_weight_high.config.yaml` | High trust influence in voting. | `trust_voting_weight: 0.9` | When does trust become harmful? |
| **E7: Knower Role** | `e7_knower.config.yaml` | One agent knows another's identity. Trust propagates this knowledge. | `enable_knower: true` | Can local knowledge become collective through trust? |

### Running Experiments

#### Single Experiment

```bash
# Run a specific experiment with default seed
python -m aegis.train.rl_ppo -c aegis/train/configs/e1_baseline.config.yaml

# Run with custom seed
python -m aegis.train.rl_ppo -c aegis/train/configs/e1_baseline.config.yaml --seed 42

# Analyze results (single seed)
python -m aegis.scripts.analyze_events checkpoints/e1_baseline/seed_42/events.jsonl --summary
```

#### Multiple Experiments with Shared Seed Set

For fair comparison (paired comparison), use the shared seed set:

```bash
# Run E1 with all seeds (200 seeds)
python -m aegis.scripts.run_experiments e1_baseline.config.yaml

# Run E1 with test seeds (first 10 seeds, for quick testing)
python -m aegis.scripts.run_experiments e1_baseline.config.yaml --test

# Run with limited seeds (for testing)
python -m aegis.scripts.run_experiments e1_baseline.config.yaml --max-seeds 3

# Run multiple experiments with all seeds
python -m aegis.scripts.run_experiments e1_baseline.config.yaml e2_fov_low.config.yaml e3_trust_comm.config.yaml

# Dry run (show commands without executing)
python -m aegis.scripts.run_experiments e1_baseline.config.yaml --dry-run
```

**Output Structure:**
Each seed creates its own subdirectory:
```
checkpoints/
  e1_baseline/
    seed_670488/
      events.jsonl
      training_log.json
      ppo_final.pt
    seed_116740/
      events.jsonl
      training_log.json
      ppo_final.pt
    ...
```

#### Merging and Analyzing Multiple Seeds

After running experiments with multiple seeds, merge events for analysis:

```bash
# Merge events from all seeds into a single file
python -m aegis.scripts.merge_events checkpoints/e1_baseline

# This creates: checkpoints/e1_baseline/events_all.jsonl

# Analyze the merged file
python -m aegis.scripts.analyze_events checkpoints/e1_baseline/events_all.jsonl --summary

# Find interesting meetings
python -m aegis.scripts.analyze_events checkpoints/e1_baseline/events_all.jsonl --find-interesting

# Analyze impostor strategies
python -m aegis.scripts.analyze_events checkpoints/e1_baseline/events_all.jsonl --impostor-strategies
```

**Why shared seed set?**
- All experiments use the same set of seeds (200 seeds by default)
- For each seed, E1, E2, E3, etc. start with the same initial world state
- This enables **paired comparison**: differences = effect of design, not luck
- Reduces variance and enables fair statistical comparison

### Fixed Parameters (Constant Across All Experiments)

- **Agents**: 4 survivors + 2 impostors
- **Map**: 9 rooms, evac_room=4
- **Vision radius**: 2 (except E2: 1 and 3)
- **Training**: 2M timesteps, PPO with same hyperparameters
- **Seeds**: Shared seed set (200 seeds) for all experiments

### Experiment Design Principles

1. **One factor per experiment**: Only the variable of interest changes
2. **Baseline first**: E1 establishes the reference point
3. **Progressive complexity**: E1 → E2 → E3 → E4–E7
4. **Statistical rigor**: Multiple runs per experiment (200 seeds) for statistical significance
5. **Paired comparison**: Same seeds across experiments enable fair comparison

## Project Structure

```
aegis/
├── core/
│   ├── world.py       # Dataclasses: WorldState, AgentState, TaskState, etc.
│   ├── rules.py       # GameConfig with all tunable parameters
│   ├── events.py      # Typed events for logging
│   └── engine.py      # Deterministic StepEngine
├── obs/
│   ├── builder.py     # ObservationBuilder for agent observations
│   └── spaces.py      # Gymnasium spaces definitions
├── comms/
│   ├── tokens.py      # Discrete token vocabulary (20 tokens)
│   └── router.py      # Message routing (broadcast/local)
├── env/
│   ├── pettingzoo_env.py  # PettingZoo ParallelEnv wrapper
│   └── wrappers.py        # SuperSuit wrappers
├── agents/
│   └── random_policy.py        # Random baseline
├── train/
│   ├── rl_ppo.py          # CleanRL PPO training script
│   └── configs/
│       └── main.config.yaml  # Main experiment configuration
├── logging/
│   ├── metrics.py         # Episode metrics
│   └── event_writer.py    # JSONL event logger
└── scripts/
    ├── analyze_events.py      # Analyze events.jsonl
    ├── merge_events.py        # Merge events from multiple seeds
    ├── run_experiments.py    # Run experiments with shared seed set
    └── evaluate_ppo.py        # Evaluate trained models
```

## Game Rules

### Win Conditions

**Survivors Win** if:
- EVAC is active (all tasks complete OR all impostors eliminated)
- At least 2 alive survivors remain in the EVAC room for 5 consecutive ticks

**Impostors Win** if:
- Alive impostors >= alive survivors
- Timeout (default 1000 ticks)

### Actions

**Survivors**:
- MOVE to any room (1 step per tick via A* pathfinding)
- WORK on tasks (when in correct room)
- REPORT bodies (triggers meeting)
- VOTE during voting phase
- SEND_TOKEN during meeting

**Impostors**:
- MOVE to any room
- KILL survivors (with cooldown)
- CLOSE_DOOR (with cooldown)
- VOTE during voting phase
- SEND_TOKEN during meeting

### Phases

1. **FREE_PLAY**: Normal gameplay, movement, tasks, kills
2. **MEETING**: 60 ticks of discussion with messaging
3. **VOTING**: 30 ticks to cast votes, majority ejects

### Tasks

- **Single-point**: Stand in target room for T ticks
- **Two-step**: Visit room A, then room B, each for T ticks

## Configuration

Key configuration options in `train/configs/base.yaml`:

```yaml
env:
  num_survivors: 4
  num_impostors: 1
  vision_radius: 2
  kill_cooldown: 30
  door_close_duration: 15
  meeting_discussion_ticks: 60
  meeting_voting_ticks: 30
  evac_ticks_required: 5
  max_ticks: 1000
  enable_knower: false
  memory_mode: none  # none, aggregated, recurrent

rewards:
  win: 1.0
  loss: -1.0
  task_step: 0.05
  correct_report: 0.2
  kill: 0.3
```

## Observation Space

Dict observation with:
- `phase`: Current game phase (0/1/2)
- `self_room`: Agent's current room
- `self_role`: Agent's role
- `evac_active`: Whether EVAC is activated
- `visible_agents`: Padded array of visible agents
- `visible_bodies`: Padded array of visible bodies
- `door_states`: State of all doors
- `task_state`: Current task info (survivors)
- `meeting_messages`: Recent messages
- `action_mask`: Valid actions mask

## Action Space

Single Discrete space with action masking:
- `MOVE_TO_ROOM[r]` for each room
- `WORK`
- `REPORT`
- `KILL[target]` for each agent (impostors only)
- `CLOSE_DOOR[door]` for each door (impostors only)
- `SEND_TOKEN[token]` for each token
- `VOTE[agent]` for each agent
- `VOTE_SKIP`

## Token Vocabulary

20 discrete tokens for communication:
- SKIP, CONFIRM, DENY
- SUSPECT_0 through SUSPECT_4
- CLEAR_0 through CLEAR_4
- BODY_IN_ROOM_0 through BODY_IN_ROOM_2
- I_WAS_IN_ROOM_0 through I_WAS_IN_ROOM_2
- DOOR_CLOSED

## Trust-Based Communication

**NEW**: Optional discrete communication system for trust experiments. See [TRUST_COMMUNICATION.md](TRUST_COMMUNICATION.md) for full documentation.

### Quick Start

```bash
# Run demo
python aegis/scripts/demo_trust_comm.py

# Run tests
pytest aegis/tests/test_trust_comm.py -v

# Train with trust communication enabled
python -m aegis.train.rllib_ppo --config aegis/train/configs/trust_comm.yaml
```

### Communication Actions

During meetings, agents can use 5 types of actions:
- `SUPPORT(agent)`: Increase trust in an agent (+0.15)
- `ACCUSE(agent)`: Decrease trust in an agent (-0.20)
- `DEFEND_SELF`: Increase self-trust (+0.05)
- `QUESTION(agent)`: Slight distrust (-0.08)
- `NO_OP`: Do nothing

### Logging & Analysis

Communication actions are logged to JSONL event streams. For experiments with multiple seeds:

```bash
# First, merge events from all seeds
python -m aegis.scripts.merge_events checkpoints/experiment_name

# Then analyze the merged file
python -m aegis.scripts.analyze_events checkpoints/experiment_name/events_all.jsonl --summary

# Find interesting meetings with coordination
python -m aegis.scripts.analyze_events checkpoints/experiment_name/events_all.jsonl --find-interesting

# Analyze impostor strategies
python -m aegis.scripts.analyze_events checkpoints/experiment_name/events_all.jsonl --impostor-strategies
```

Example event structure:
```json
{
  "event_type": "comm_action",
  "tick": 123,
  "sender_id": 2,
  "action_id": 7,
  "action_name": "ACCUSE(1)"
}
```

### Key Configuration

```yaml
enable_trust_comm: true           # Enable/disable system
trust_delay_ticks: 5              # Delay before trust updates
trust_voting_weight: 0.3          # Weight in voting (0=off, 1=full)
```

### Ablation Studies

See the [Experiments](#experiments) section above for the complete experimental design. Key ablations include:

- **Communication On/Off**: E1 (off) vs E3 (on)
- **Trust Weight**: E5 series (0.0, 0.3, 0.9)
- **Delay Sensitivity**: E4 series (immediate vs delayed)
- **Field of View**: E6 series (low vs high visibility)

## Memory Modes

- **none**: Observation only, no memory
- **aggregated**: Last-seen tracking in observation
- **recurrent**: Use LSTM model (if implemented)

## License

MIT License

## Author
Marshanishvili Mykhailo

