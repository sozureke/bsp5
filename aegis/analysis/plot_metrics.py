"""Generate bar plots for experiment metrics."""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Consistent color scheme
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Purple
    'accent': '#F18F01',       # Orange
    'success': '#06A77D',      # Green
    'warning': '#D90429',      # Red
}

# Experiment display names (for better readability)
EXPERIMENT_NAMES = {
    'e1_baseline': 'E1: Baseline',
    'e2_fov_high': 'E2: FoV High',
    'e2_fov_low': 'E2: FoV Low',
    'e3_trust_comm': 'E3: Trust + Comm',
    'e4_delay_delayed': 'E4: Delay (Delayed)',
    'e4_delay_immediate': 'E4: Delay (Immediate)',
}


def load_metrics(metrics_path: Path) -> Dict:
    """Load metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def get_experiment_order(metrics: Dict) -> List[str]:
    """Get consistent experiment ordering (alphabetical by experiment name)."""
    return sorted(metrics.keys())


def extract_plot_data(metrics: Dict, experiment_order: List[str]) -> Dict:
    """Extract data for plotting."""
    data = {
        'experiments': [],
        'survivor_win_rate': [],
        'ejection_rate_pct': [],
        'correct_ejection_rate_pct': [],
        'avg_episode_length': [],
        'avg_comm_per_meeting': [],
        'win_conditions': {},  # For stacked bar chart
        'impostor_ejection_per_game': [],  # Impostor ejection rate per game
        'comm_by_role_type': {},  # Communication by role and action type
    }
    
    for exp_name in experiment_order:
        exp_data = metrics[exp_name]
        data['experiments'].append(EXPERIMENT_NAMES.get(exp_name, exp_name))
        data['survivor_win_rate'].append(exp_data['wins']['survivor_rate'])
        data['ejection_rate_pct'].append(exp_data['meetings']['ejection_rate_pct'])
        data['correct_ejection_rate_pct'].append(exp_data['meetings']['correct_ejection_rate_pct'])
        data['avg_episode_length'].append(exp_data['episodes']['avg_length_ticks'])
        data['avg_comm_per_meeting'].append(exp_data['communication']['avg_per_meeting'])
        
        # Win conditions breakdown
        win_reasons = exp_data['wins']['win_reasons']
        total_games = exp_data['total_games']
        
        # Survivor win conditions
        survivor_evac = win_reasons.get('evac_success', 0)
        survivor_eliminated = win_reasons.get('all_impostors_eliminated', 0)
        
        # Impostor win conditions
        impostor_parity = win_reasons.get('impostor_parity', 0)
        impostor_all_dead = win_reasons.get('all_survivors_dead', 0)
        impostor_timeout = win_reasons.get('timeout', 0)
        
        data['win_conditions'][exp_name] = {
            'survivor_evac': (survivor_evac / total_games * 100) if total_games > 0 else 0,
            'survivor_eliminated': (survivor_eliminated / total_games * 100) if total_games > 0 else 0,
            'impostor_parity': (impostor_parity / total_games * 100) if total_games > 0 else 0,
            'impostor_all_dead': (impostor_all_dead / total_games * 100) if total_games > 0 else 0,
            'impostor_timeout': (impostor_timeout / total_games * 100) if total_games > 0 else 0,
        }
        
        # Impostor ejection rate per game (approximation: correct ejections / total games)
        correct_ejections = exp_data['meetings']['correct_ejections']
        impostor_ejection_rate = (correct_ejections / total_games * 100) if total_games > 0 else 0
        data['impostor_ejection_per_game'].append(impostor_ejection_rate)
        
        # Communication by role and type (for selected experiments)
        if exp_name in ['e2_fov_low', 'e3_trust_comm', 'e4_delay_delayed']:
            # Use detailed breakdown if available, otherwise estimate
            if 'by_role_type' in exp_data['communication']:
                comm_by_role_type = exp_data['communication']['by_role_type']
                data['comm_by_role_type'][exp_name] = {
                    'survivor': comm_by_role_type.get('survivor', {}),
                    'impostor': comm_by_role_type.get('impostor', {}),
                }
            else:
                # Fallback: estimate from totals
                comm_by_type = exp_data['communication']['by_type']
                comm_by_role = exp_data['communication']['by_role']
                
                total_survivor = comm_by_role.get('survivor', 0)
                total_impostor = comm_by_role.get('impostor', 0)
                total_comm = total_survivor + total_impostor
                
                survivor_actions = {}
                impostor_actions = {}
                
                if total_comm > 0:
                    survivor_ratio = total_survivor / total_comm
                    impostor_ratio = total_impostor / total_comm
                    
                    action_categories = {
                        'ACCUSE': 0,
                        'SUPPORT': 0,
                        'QUESTION': 0,
                        'DEFEND_SELF': 0,
                        'NO_OP': 0,
                    }
                    
                    for action_name, count in comm_by_type.items():
                        if action_name.startswith('ACCUSE'):
                            action_categories['ACCUSE'] += count
                        elif action_name.startswith('SUPPORT'):
                            action_categories['SUPPORT'] += count
                        elif action_name.startswith('QUESTION'):
                            action_categories['QUESTION'] += count
                        elif action_name == 'DEFEND_SELF':
                            action_categories['DEFEND_SELF'] += count
                        elif action_name == 'NO_OP':
                            action_categories['NO_OP'] += count
                    
                    for category, count in action_categories.items():
                        survivor_actions[category] = count * survivor_ratio
                        impostor_actions[category] = count * impostor_ratio
                
                data['comm_by_role_type'][exp_name] = {
                    'survivor': survivor_actions,
                    'impostor': impostor_actions,
                }
    
    return data


def create_bar_plot(
    x_labels: List[str],
    y_values: List[float],
    title: str,
    ylabel: str,
    color: str,
    output_path: Path,
    figsize: Tuple[int, int] = (10, 6),
    ylim: Tuple[float, float] = None,
):
    """Create a single bar plot."""
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(
        x_labels,
        y_values,
        color=color,
        edgecolor='white',
        linewidth=1.5,
        alpha=0.8,
    )
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.1f}',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
        )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Experiment', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Set y-axis limits if provided
    if ylim:
        ax.set_ylim(ylim)
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def create_stacked_bar_plot(
    x_labels: List[str],
    data_dict: Dict[str, Dict[str, float]],
    title: str,
    ylabel: str,
    output_path: Path,
    figsize: Tuple[int, int] = (12, 6),
    ylim: Tuple[float, float] = None,
):
    """Create a stacked bar chart for win conditions."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors for each segment
    segment_colors = {
        'survivor_evac': '#06A77D',  # Green
        'survivor_eliminated': '#4ECDC4',  # Light green
        'impostor_parity': '#D90429',  # Red
        'impostor_all_dead': '#FF6B6B',  # Light red
        'impostor_timeout': '#FFA07A',  # Light orange
    }
    
    segment_labels = {
        'survivor_evac': 'Survivor: Evac',
        'survivor_eliminated': 'Survivor: Eliminated',
        'impostor_parity': 'Impostor: Parity',
        'impostor_all_dead': 'Impostor: All Dead',
        'impostor_timeout': 'Impostor: Timeout',
    }
    
    # Prepare data for stacking
    segments = ['survivor_evac', 'survivor_eliminated', 'impostor_parity', 'impostor_all_dead', 'impostor_timeout']
    
    bottom = np.zeros(len(x_labels))
    
    for segment in segments:
        values = [data_dict.get(exp, {}).get(segment, 0) for exp in sorted(data_dict.keys())]
        ax.bar(
            x_labels,
            values,
            bottom=bottom,
            label=segment_labels[segment],
            color=segment_colors[segment],
            edgecolor='white',
            linewidth=1.5,
            alpha=0.8,
        )
        bottom += values
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Experiment', fontsize=12)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Set y-axis limits
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0, 100)
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def create_grouped_bar_plot(
    x_labels: List[str],
    data_dict: Dict[str, Dict[str, Dict[str, float]]],
    title: str,
    ylabel: str,
    output_path: Path,
    figsize: Tuple[int, int] = (12, 6),
):
    """Create a grouped bar chart for communication composition."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Action categories
    categories = ['ACCUSE', 'SUPPORT', 'QUESTION', 'DEFEND_SELF', 'NO_OP']
    category_colors = {
        'ACCUSE': '#D90429',  # Red
        'SUPPORT': '#06A77D',  # Green
        'QUESTION': '#2E86AB',  # Blue
        'DEFEND_SELF': '#F18F01',  # Orange
        'NO_OP': '#A23B72',  # Purple
    }
    
    x = np.arange(len(x_labels))
    width = 0.35  # Width of bars
    
    # Calculate positions for grouped bars
    n_categories = len(categories)
    bar_width = width / n_categories
    
    for i, category in enumerate(categories):
        survivor_values = []
        impostor_values = []
        
        for exp_name in sorted(data_dict.keys()):
            exp_data = data_dict[exp_name]
            survivor_values.append(exp_data['survivor'].get(category, 0))
            impostor_values.append(exp_data['impostor'].get(category, 0))
        
        offset = (i - n_categories / 2) * bar_width + bar_width / 2
        
        ax.bar(
            x + offset - width/2,
            survivor_values,
            bar_width,
            label=f'{category} (Survivor)' if i == 0 else '',
            color=category_colors[category],
            alpha=0.7,
            edgecolor='white',
            linewidth=0.5,
        )
        
        ax.bar(
            x + offset + width/2,
            impostor_values,
            bar_width,
            label=f'{category} (Impostor)' if i == 0 else '',
            color=category_colors[category],
            alpha=0.4,
            edgecolor='white',
            linewidth=0.5,
            hatch='///',
        )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=category_colors[cat], alpha=0.7, label=f'{cat} (Survivor)')
        for cat in categories
    ] + [
        Patch(facecolor=category_colors[cat], alpha=0.4, hatch='///', label=f'{cat} (Impostor)')
        for cat in categories
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def main():
    metrics_path = Path("results/all_metrics.json")
    
    if not metrics_path.exists():
        print(f"Error: {metrics_path} not found")
        print("Run: python -m aegis.analysis.aggregate_all_metrics")
        return 1
    
    print(f"Loading metrics from {metrics_path}...")
    metrics = load_metrics(metrics_path)
    
    experiment_order = get_experiment_order(metrics)
    print(f"Found {len(experiment_order)} experiments: {', '.join(experiment_order)}")
    
    data = extract_plot_data(metrics, experiment_order)
    
    # Create output directory
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating plots...")
    
    # 1. Survivor win rate
    create_bar_plot(
        x_labels=data['experiments'],
        y_values=data['survivor_win_rate'],
        title='Survivor Win Rate by Experiment',
        ylabel='Win Rate (%)',
        color=COLORS['success'],
        output_path=output_dir / 'survivor_win_rate.png',
        ylim=(0, 100),
    )
    
    # 2. Percentage of meetings with ejection
    create_bar_plot(
        x_labels=data['experiments'],
        y_values=data['ejection_rate_pct'],
        title='Percentage of Meetings with Ejection',
        ylabel='Ejection Rate (%)',
        color=COLORS['primary'],
        output_path=output_dir / 'ejection_rate.png',
        ylim=(0, 100),
    )
    
    # 3. Correct ejection rate
    create_bar_plot(
        x_labels=data['experiments'],
        y_values=data['correct_ejection_rate_pct'],
        title='Correct Ejection Rate',
        ylabel='Correct Ejection Rate (%)',
        color=COLORS['accent'],
        output_path=output_dir / 'correct_ejection_rate.png',
        ylim=(0, 100),
    )
    
    # 4. Average episode length
    create_bar_plot(
        x_labels=data['experiments'],
        y_values=data['avg_episode_length'],
        title='Average Episode Length',
        ylabel='Average Length (ticks)',
        color=COLORS['secondary'],
        output_path=output_dir / 'avg_episode_length.png',
    )
    
    # 5. Communication actions per meeting (optional)
    create_bar_plot(
        x_labels=data['experiments'],
        y_values=data['avg_comm_per_meeting'],
        title='Average Communication Actions per Meeting',
        ylabel='Actions per Meeting',
        color=COLORS['warning'],
        output_path=output_dir / 'avg_comm_per_meeting.png',
    )
    
    # 6. Win Conditions Breakdown (Stacked Bar) - REQUIRED
    win_conditions_dict = {}
    for exp_name in experiment_order:
        if exp_name in data['win_conditions']:
            win_conditions_dict[exp_name] = data['win_conditions'][exp_name]
    
    create_stacked_bar_plot(
        x_labels=[EXPERIMENT_NAMES.get(exp, exp) for exp in experiment_order],
        data_dict=win_conditions_dict,
        title='Win Conditions Breakdown by Experiment',
        ylabel='Percentage of Games (%)',
        output_path=output_dir / 'win_conditions_breakdown.png',
        ylim=(0, 100),
    )
    
    # 7. Impostor Ejection Rate per Game - REQUIRED
    create_bar_plot(
        x_labels=data['experiments'],
        y_values=data['impostor_ejection_per_game'],
        title='Impostor Ejection Rate per Game',
        ylabel='Ejection Rate per Game (%)',
        color=COLORS['secondary'],
        output_path=output_dir / 'impostor_ejection_per_game.png',
    )
    
    # 8. Communication Composition (Role × Action Type) - REQUIRED for selected experiments
    if data['comm_by_role_type']:
        comm_experiments = []
        comm_data = {}
        
        for exp_name in ['e2_fov_low', 'e3_trust_comm', 'e4_delay_delayed']:
            if exp_name in data['comm_by_role_type']:
                comm_experiments.append(EXPERIMENT_NAMES.get(exp_name, exp_name))
                comm_data[exp_name] = data['comm_by_role_type'][exp_name]
        
        if comm_experiments:
            create_grouped_bar_plot(
                x_labels=comm_experiments,
                data_dict=comm_data,
                title='Communication Composition by Role and Action Type',
                ylabel='Number of Actions',
                output_path=output_dir / 'comm_composition_by_role_type.png',
            )
    
    print(f"\n✓ All plots saved to {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
