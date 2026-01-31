#!/usr/bin/env python3
"""
Generate figures for the AgentBusters FAB++ paper.
Based on evaluation results from 2026-01-31.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# Results data from evaluation run yxc20089-20260131-154253
RESULTS = {
    'overall': 69.5,
    'sections': {
        'Knowledge Retrieval': {'score': 66.7, 'tasks': 6, 'weight': 0.20},
        'Analytical Reasoning': {'score': 100.0, 'tasks': 3, 'weight': 0.20},
        'Professional Tasks': {'score': 76.5, 'tasks': 4, 'weight': 0.20},
        'Options Trading': {'score': 61.2, 'tasks': 3, 'weight': 0.20},
        'Crypto Trading': {'score': 43.0, 'tasks': 2, 'weight': 0.20},
    },
    'gdpval_subscores': {
        'Completion': 19.0,
        'Accuracy': 19.8,
        'Format': 18.2,
        'Professionalism': 19.5,
    },
    'options_subscores': {
        'Strategy': 75.0,
        'P&L': 86.7,
        'Risk': 63.3,
        'Greeks': 20.0,
    },
    'crypto_subscores': {
        'Baseline': 43.8,
        'Noisy': 44.3,
        'Meta': 45.5,
        'Adversarial': 39.3,
    },
}


def fig0_system_architecture():
    """System architecture diagram - clean professional design."""
    from matplotlib.patches import FancyBboxPatch, ConnectionPatch

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Colors - softer, more professional palette
    green_color = '#4CAF50'
    purple_color = '#9C27B0'
    mcp_color = '#2196F3'
    text_dark = '#212121'
    border_color = '#757575'

    # ===== TOP ROW: Agents =====
    agent_y = 4.2
    agent_height = 1.4
    agent_width = 3.2

    # Green Agent
    green_box = FancyBboxPatch((0.8, agent_y), agent_width, agent_height,
                                boxstyle="round,pad=0.02,rounding_size=0.15",
                                facecolor=green_color, edgecolor='#388E3C', linewidth=2)
    ax.add_patch(green_box)
    ax.text(0.8 + agent_width/2, agent_y + 1.05, 'Green Agent', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')
    ax.text(0.8 + agent_width/2, agent_y + 0.65, 'Evaluator', ha='center', va='center',
            fontsize=11, color='white', alpha=0.9)
    ax.text(0.8 + agent_width/2, agent_y + 0.25, 'Task Generation & Scoring', ha='center', va='center',
            fontsize=9, color='white', alpha=0.8)

    # Purple Agent
    purple_x = 7
    purple_box = FancyBboxPatch((purple_x, agent_y), agent_width, agent_height,
                                 boxstyle="round,pad=0.02,rounding_size=0.15",
                                 facecolor=purple_color, edgecolor='#7B1FA2', linewidth=2)
    ax.add_patch(purple_box)
    ax.text(purple_x + agent_width/2, agent_y + 1.05, 'Purple Agent', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')
    ax.text(purple_x + agent_width/2, agent_y + 0.65, 'Finance Analyst', ha='center', va='center',
            fontsize=11, color='white', alpha=0.9)
    ax.text(purple_x + agent_width/2, agent_y + 0.25, 'LLM-Powered Analysis', ha='center', va='center',
            fontsize=9, color='white', alpha=0.8)

    # A2A Protocol connection
    arrow_y = agent_y + agent_height/2
    ax.annotate('', xy=(purple_x - 0.15, arrow_y), xytext=(0.8 + agent_width + 0.15, arrow_y),
                arrowprops=dict(arrowstyle='<->', color=text_dark, lw=2,
                               shrinkA=0, shrinkB=0))
    ax.text(5.5, arrow_y + 0.35, 'A2A Protocol', ha='center', va='center', fontsize=10,
            fontweight='bold', color=text_dark)

    # ===== BOTTOM ROW: MCP Servers in container =====
    container_x = 0.5
    container_width = 10
    container_y = 0.4
    container_height = 1.8

    # Container box
    mcp_container = FancyBboxPatch((container_x, container_y), container_width, container_height,
                                    boxstyle="round,pad=0.02,rounding_size=0.15",
                                    facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2)
    ax.add_patch(mcp_container)

    # MCP label at top of container
    ax.text(container_x + container_width/2, container_y + container_height - 0.25,
            'MCP Servers', ha='center', va='center',
            fontsize=12, fontweight='bold', color='#1565C0')

    # Individual MCP server boxes inside container - with better padding
    mcp_y = 0.55
    mcp_height = 0.95
    mcp_width = 1.45
    mcp_gap = 0.12
    # Calculate start_x to center all boxes with equal padding on sides
    total_mcp_width = 6 * mcp_width + 5 * mcp_gap
    mcp_start_x = container_x + (container_width - total_mcp_width) / 2

    mcp_servers = [
        ('SEC EDGAR', '10-K/10-Q Filings'),
        ('Yahoo Finance', 'Market Data'),
        ('Sandbox', 'Code Execution'),
        ('Options', 'Greeks & IV'),
        ('Trading Sim', 'Paper Trading'),
        ('Risk', 'VaR & Stress'),
    ]

    for i, (name, desc) in enumerate(mcp_servers):
        x = mcp_start_x + i * (mcp_width + mcp_gap)
        mcp_box = FancyBboxPatch((x, mcp_y), mcp_width, mcp_height,
                                  boxstyle="round,pad=0.02,rounding_size=0.1",
                                  facecolor=mcp_color, edgecolor='#1976D2', linewidth=1.5)
        ax.add_patch(mcp_box)
        ax.text(x + mcp_width/2, mcp_y + 0.6, name, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')
        ax.text(x + mcp_width/2, mcp_y + 0.28, desc, ha='center', va='center',
                fontsize=7, color='white', alpha=0.9)

    # Connection arrows from agents to MCP container (double-headed)
    arrow_color = '#757575'
    green_center_x = 0.8 + agent_width/2
    purple_center_x = purple_x + agent_width/2
    container_top = container_y + container_height

    # Double-headed arrow from Green Agent to container
    ax.annotate('', xy=(green_center_x, container_top + 0.05),
                xytext=(green_center_x, agent_y - 0.05),
                arrowprops=dict(arrowstyle='<->', color=arrow_color, lw=2))
    # MCP label on green arrow
    mid_y_green = (agent_y + container_top) / 2
    ax.text(green_center_x + 0.35, mid_y_green, 'MCP', ha='left', va='center', fontsize=10,
            fontweight='bold', color=arrow_color)

    # Double-headed arrow from Purple Agent to container
    ax.annotate('', xy=(purple_center_x, container_top + 0.05),
                xytext=(purple_center_x, agent_y - 0.05),
                arrowprops=dict(arrowstyle='<->', color=arrow_color, lw=2))
    # MCP label on purple arrow
    ax.text(purple_center_x + 0.35, mid_y_green, 'MCP', ha='left', va='center', fontsize=10,
            fontweight='bold', color=arrow_color)

    plt.tight_layout()
    plt.savefig('figures/system_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/system_architecture.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/system_architecture.pdf")


def fig1_section_scores():
    """Bar chart comparing section scores."""
    sections = list(RESULTS['sections'].keys())
    scores = [RESULTS['sections'][s]['score'] for s in sections]
    tasks = [RESULTS['sections'][s]['tasks'] for s in sections]

    # Short names for x-axis
    short_names = ['Knowledge', 'Analysis', 'GDPVal', 'Options', 'Crypto']

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    bars = ax.bar(short_names, scores, color=colors, edgecolor='black', linewidth=1.2)

    # Add score labels on bars
    for bar, score, task in zip(bars, scores, tasks):
        height = bar.get_height()
        ax.annotate(f'{score:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.annotate(f'({task} tasks)',
                    xy=(bar.get_x() + bar.get_width() / 2, height/2),
                    ha='center', va='center', fontsize=9, color='white')

    # Add overall score line
    ax.axhline(y=RESULTS['overall'], color='red', linestyle='--', linewidth=2,
               label=f"Overall: {RESULTS['overall']:.1f}")

    ax.set_ylabel('Score (0-100)')
    ax.set_title('FAB++ Section Scores (GPT-4o Baseline)')
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('figures/section_scores.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/section_scores.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/section_scores.pdf")


def fig2_gdpval_breakdown():
    """GDPVal sub-scores breakdown (out of 25 each)."""
    categories = list(RESULTS['gdpval_subscores'].keys())
    scores = list(RESULTS['gdpval_subscores'].values())
    max_score = 25

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(categories))
    width = 0.6

    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    bars = ax.bar(x, scores, width, color=colors, edgecolor='black', linewidth=1.2)

    # Add percentage labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        pct = (score / max_score) * 100
        ax.annotate(f'{score:.1f}/25\n({pct:.0f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Score')
    ax.set_title('GDPVal Professional Tasks: Sub-score Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 30)
    ax.axhline(y=max_score, color='gray', linestyle=':', alpha=0.5, label='Max Score (25)')
    ax.legend()

    plt.tight_layout()
    plt.savefig('figures/gdpval_breakdown.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/gdpval_breakdown.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/gdpval_breakdown.pdf")


def fig3_crypto_transforms():
    """Crypto trading scores across different transforms."""
    transforms = list(RESULTS['crypto_subscores'].keys())
    scores = list(RESULTS['crypto_subscores'].values())

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#27ae60', '#3498db', '#9b59b6', '#e74c3c']
    bars = ax.bar(transforms, scores, color=colors, edgecolor='black', linewidth=1.2)

    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.annotate(f'{score:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Score (0-100)')
    ax.set_title('Crypto Trading: Performance Across Market Conditions')
    ax.set_ylim(0, 60)

    # Add annotations explaining transforms
    ax.annotate('Clean data', xy=(0, 5), ha='center', fontsize=8, style='italic')
    ax.annotate('+Price noise', xy=(1, 5), ha='center', fontsize=8, style='italic')
    ax.annotate('Combined', xy=(2, 5), ha='center', fontsize=8, style='italic')
    ax.annotate('+Signal injection', xy=(3, 5), ha='center', fontsize=8, style='italic')

    plt.tight_layout()
    plt.savefig('figures/crypto_transforms.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/crypto_transforms.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/crypto_transforms.pdf")


def fig4_options_breakdown():
    """Options trading sub-scores."""
    categories = list(RESULTS['options_subscores'].keys())
    scores = list(RESULTS['options_subscores'].values())

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    bars = ax.bar(categories, scores, color=colors, edgecolor='black', linewidth=1.2)

    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.annotate(f'{score:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Score (0-100)')
    ax.set_title('Options Trading: Four-Dimensional Breakdown')
    ax.set_ylim(0, 100)
    ax.axhline(y=RESULTS['sections']['Options Trading']['score'],
               color='red', linestyle='--', linewidth=2,
               label=f"Section Average: {RESULTS['sections']['Options Trading']['score']:.1f}")
    ax.legend()

    plt.tight_layout()
    plt.savefig('figures/options_breakdown.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/options_breakdown.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/options_breakdown.pdf")


def fig5_radar_chart():
    """Radar chart showing capability profile."""
    categories = ['Knowledge', 'Analysis', 'GDPVal', 'Options', 'Crypto']
    scores = [
        RESULTS['sections']['Knowledge Retrieval']['score'],
        RESULTS['sections']['Analytical Reasoning']['score'],
        RESULTS['sections']['Professional Tasks']['score'],
        RESULTS['sections']['Options Trading']['score'],
        RESULTS['sections']['Crypto Trading']['score'],
    ]

    # Number of variables
    num_vars = len(categories)

    # Compute angle for each category
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Complete the loop
    scores += scores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Draw the outline
    ax.plot(angles, scores, 'o-', linewidth=2, color='#3498db')
    ax.fill(angles, scores, alpha=0.25, color='#3498db')

    # Fix axis to go from 0 to 100
    ax.set_ylim(0, 100)

    # Draw labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)

    # Add score labels at each point
    for angle, score, cat in zip(angles[:-1], scores[:-1], categories):
        ax.annotate(f'{score:.1f}', xy=(angle, score), xytext=(angle, score + 8),
                    ha='center', fontsize=10, fontweight='bold')

    ax.set_title('FAB++ Capability Profile\n(GPT-4o Baseline)', size=14, y=1.08)

    plt.tight_layout()
    plt.savefig('figures/radar_chart.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/radar_chart.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/radar_chart.pdf")


def fig6_weight_distribution():
    """Pie chart showing section weight distribution."""
    sections = ['Knowledge\nRetrieval', 'Analytical\nReasoning', 'Professional\nTasks (GDPVal)',
                'Options\nTrading', 'Crypto\nTrading']
    weights = [20, 20, 20, 20, 20]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']

    fig, ax = plt.subplots(figsize=(8, 8))

    wedges, texts, autotexts = ax.pie(weights, labels=sections, autopct='%1.0f%%',
                                       colors=colors, startangle=90,
                                       wedgeprops=dict(edgecolor='black', linewidth=1.5))

    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')

    ax.set_title('FAB++ Section Weight Distribution', size=14)

    plt.tight_layout()
    plt.savefig('figures/weight_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/weight_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/weight_distribution.pdf")


if __name__ == '__main__':
    import os
    os.chdir('/Users/berta/Projects/AgentBusters/paper')

    print("Generating FAB++ paper figures...")
    fig0_system_architecture()
    fig1_section_scores()
    fig2_gdpval_breakdown()
    fig3_crypto_transforms()
    fig4_options_breakdown()
    fig5_radar_chart()
    fig6_weight_distribution()
    print("\nAll figures generated successfully!")
