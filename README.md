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

Communication actions are logged to JSONL event streams:

```bash
# Analyze communication patterns and meetings from logs
python -m aegis.scripts.analyze_events checkpoints/experiment_name/events.jsonl --summary

# Find interesting meetings with coordination
python -m aegis.scripts.analyze_events checkpoints/experiment_name/events.jsonl --find-interesting

# Analyze impostor strategies
python -m aegis.scripts.analyze_events checkpoints/experiment_name/events.jsonl --impostor-strategies
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

- **Communication On/Off**: Compare with and without communication
- **Trust Weight**: Vary from 0.0 (pure suspicion) to 1.0 (pure trust)
- **Delay Sensitivity**: Test immediate vs delayed evidence
- **Trust Deltas**: Tune strength of communication effects

## Memory Modes

- **none**: Observation only, no memory
- **aggregated**: Last-seen tracking in observation
- **recurrent**: Use LSTM model (if implemented)

## License

MIT License

## Author
Marshanishvili Mykhailo

