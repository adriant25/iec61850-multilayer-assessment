# -*- coding: utf-8 -*-
"""
simulation.py -- Main simulation loop, Excel expo     rts, and figures.

IEC 61850 Digital Substation -- Complex Network Scalability Simulation
----------------------------------------------------------------------
Entry point for the full scalability study.  Running this file executes:

  1. run_simulation()            -- Sweep N = 1 … max_bays bays across all
                                   GOOSE burst snapshots for one scenario.
  2. export_organized_report()   -- Multi-sheet Excel report (Global_Report.xlsx).
  3. export_metrics_by_scenario()-- Per-scenario Excel (Metrics_By_Scenario.xlsx).
  4. All plot_*() functions       -- Eight publication-quality figures saved to
                                   the results/ directory at 300 dpi.

Usage
-----
    python simulation.py

All outputs are written to  results/  (created automatically on first run).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from config  import T_MAX
from model   import (generate_goose_burst_timeline, generate_devices,
                     generate_demand_tensor, update_goose_demand, build_topology)
from analysis import (compute_weights, analyze_network, compute_e2e_metrics,
                      compute_cbr, compute_recovery_time)

# Ensure the output directory exists before any file is written
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def _results(filename: str) -> str:
    """Return the full path for an output file inside results/."""
    return os.path.join(RESULTS_DIR, filename)


# =============================================================================
# 1. MAIN SIMULATION LOOP
# =============================================================================

def run_simulation(max_bays: int = 10, scenario: str = 'base') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the scalability simulation for N = 1 … max_bays bays.

    For each bay count N, the function:
      1. Generates the device list and switch assignment.
      2. Builds the 5-layer topology (topology is static within one N).
      3. Sweeps over all GOOSE burst time snapshots, updating only the
         GOOSE VLANs at each step (all other VLANs are re-used by reference).
      4. Computes physical weights (latency, jitter, PLR) for each snapshot.
      5. Derives all graph-theoretic and QoS metrics.
      6. Stores per-node centrality in long format and vulnerability at
         steady state.

    Args:
        max_bays: Maximum number of bays to simulate (default 10).
        scenario: 'base' (all 100 Mbps), 'upgraded' (trunk + BBP link 1 Gbps) or 'bbp_only' (BBP link 1 Gbps).

    Returns:
        df_main: Long-format DataFrame -- one row per (N, scenario, snapshot, node).
        df_vuln: DataFrame with node vulnerability indices (steady state only).
    """
    history      = []  # One dict per (N, snapshot, node)
    vuln_history = []  # One dict per (N, node) -- only at steady state

    print(f"Starting simulation: N=1 to {max_bays} bays  [scenario: {scenario}]")

    t_snapshots, delta_t_list = generate_goose_burst_timeline()

    for n_bays in range(1, max_bays + 1):
        print(f"  Simulating N={n_bays} ...")

        # Build the static model once per bay count
        devices, sw_map = generate_devices(n_bays)
        D_base          = generate_demand_tensor(n_bays, devices, delta_t=1000.0)
        labels, A       = build_topology(n_bays, devices, D_base, sw_map)

        for t, delta_t in zip(t_snapshots, delta_t_list):
            print(f"    -> t={t:.3f} ms  dt={delta_t:.3f} ms")

            # Update only GOOSE VLANs (V3/V4) -- all other DFs re-used by reference
            D = update_goose_demand(D_base, n_bays, delta_t)

            # Compute physical weights for this snapshot
            W, J, PL, u_p = compute_weights(
                labels, A, D, sw_map, devices, delta_t, scenario
            )

            # Derive all metrics
            # Approximate betweenness for large graphs (N >= 6) to keep runtime tractable
            use_approx = (n_bays >= 6)
            # Vulnerability index is expensive: compute only at steady state
            is_steady  = (delta_t == 1000.0)
            metrics = analyze_network(
                n_bays, labels, W, J, PL, u_p,
                use_approx=use_approx,
                compute_vulnerability=is_steady,
            )

            # End-to-end SV / GOOSE flow latency (path sums) -- the quantity the DES
            # measures. IEC compliance is judged on these, not on single edges.
            e2e = compute_e2e_metrics(labels, W, PL, D, sw_map, scenario,
                                      D_window=D_base,
                                      synchronized_burst=(delta_t < 1000.0))
            metrics.update(e2e)
            metrics['IEC_Violation'] = int(
                e2e['E2E_Max_SV_us'] >= T_MAX * 1e6
                or e2e['E2E_Max_GOOSE_us'] >= T_MAX * 1e6
            )

            # Pop internal dicts before they are stored as flat records
            centrality_dict = metrics.pop('centrality_dict')
            vuln_dict       = metrics.pop('vuln_dict')

            # Save vulnerability records (steady state only -- one entry per node)
            for node_label, v_score in vuln_dict.items():
                vuln_history.append({
                    'N_Bays':             n_bays,
                    'Scenario':           scenario,
                    'Node_Label':         node_label,
                    'Vulnerability_Index': v_score,
                    'Betweenness':        centrality_dict.get(node_label, 0.0),
                })

            # Base record shared across all nodes at this snapshot
            base_record = {
                'N_Bays':          n_bays,
                'Scenario':        scenario,
                'Snapshot_Time_ms': t,
                'Delta_T_ms':      delta_t,
                **metrics,
            }

            if not centrality_dict:
                base_record['Node_Label'] = None
                base_record['Centrality'] = 0.0
                history.append(base_record)
            else:
                for node, centrality in centrality_dict.items():
                    record = base_record.copy()
                    record['Node_Label'] = node
                    record['Centrality'] = centrality
                    history.append(record)

    return pd.DataFrame(history), pd.DataFrame(vuln_history)


# =============================================================================
# 2. EXCEL EXPORTS
# =============================================================================

def export_organized_report(
    df: pd.DataFrame,
    df_vuln: pd.DataFrame,
    filename: str = 'Global_Report.xlsx',
) -> None:
    """
    Export a comprehensive multi-sheet Excel workbook.

    Sheets:
      - Executive_Summary : KPI averages + IEC 61850 compliance status per N
      - Graph_Metrics     : λ₂, Global Efficiency, Gini vs N (snapshot average)
      - Recovery_Time     : GOOSE burst recovery time per (N, scenario)
      - Vulnerability     : Node Vulnerability Index (steady state)
      - CBR_Top_Nodes     : Top-30 nodes by Centrality Burst Ratio at max N
      - Raw_Data          : Complete snapshot-level dataset

    Args:
        df:       Main simulation DataFrame.
        df_vuln:  Vulnerability DataFrame.
        filename: Output filename (written to results/).
    """
    filepath = _results(filename)
    print(f"Generating global report -> {filepath}")

    df_snap = df.drop(columns=['Node_Label', 'Centrality']).drop_duplicates().copy()
    avg_df  = df_snap.groupby(['N_Bays', 'Scenario']).mean(numeric_only=True).reset_index()
    max_df  = df_snap.groupby(['N_Bays', 'Scenario']).max(numeric_only=True).reset_index()

    # --- Executive Summary ---
    summary = pd.DataFrame({
        'N_Bays':                     avg_df['N_Bays'],
        'Scenario':                   avg_df['Scenario'],
        'E2E Avg SV (us)':            avg_df.get('E2E_Avg_SV_us',     pd.Series(dtype=float)),
        'E2E Max SV (us, worst snapshot)':    max_df.get('E2E_Max_SV_us',    pd.Series(dtype=float)),
        'E2E Avg GOOSE (us)':         avg_df.get('E2E_Avg_GOOSE_us',  pd.Series(dtype=float)),
        'E2E Max GOOSE (us, worst snapshot)': max_df.get('E2E_Max_GOOSE_us', pd.Series(dtype=float)),
        'E2E Max SV PLR (%)':         max_df.get('E2E_PLR_Max_SV',    pd.Series(dtype=float)),
        'Queue wait Avg P6-Prot (ms)':   avg_df.get('Latency_Avg_P6',    pd.Series(dtype=float)),
        'Queue wait Max P4-SV (ms)':     avg_df.get('Latency_Max_P4',    pd.Series(dtype=float)),
        'Global Efficiency Avg':      avg_df.get('Global_Efficiency', pd.Series(dtype=float)),
        'Load Gini Avg':              avg_df.get('Load_Gini',         pd.Series(dtype=float)),
        'Fiedler λ₂ Avg':             avg_df.get('Fiedler_Lambda2',   pd.Series(dtype=float)),
        'Max Switch Util (%)': (
            max_df[[c for c in max_df.columns if 'Max_Util' in c]].max(axis=1) * 100
        ),
    })

    fail_sum = (df_snap.groupby(['N_Bays', 'Scenario'])['IEC_Violation']
                .sum().reset_index()
                .rename(columns={'IEC_Violation': 'Violation_Snapshots'}))
    summary = pd.merge(summary, fail_sum, on=['N_Bays', 'Scenario'])

    def check_compliance(group: pd.DataFrame) -> str:
        if group['IEC_Violation'].sum() > 0:
            return 'FAIL (Latency > 3 ms)'
        if group['Loss_Avg_P6'].max() > 0.01:
            return 'FAIL (Packet Loss)'
        return 'PASS'

    compliance = (df_snap.groupby(['N_Bays', 'Scenario'])
                  .apply(check_compliance)
                  .rename('IEC61850_Status')
                  .reset_index())
    summary = pd.merge(summary, compliance, on=['N_Bays', 'Scenario'])

    # --- Graph Metrics ---
    graph_cols = ['N_Bays', 'Scenario', 'Fiedler_Lambda2',
                  'Global_Efficiency', 'Load_Gini', 'Total_Nodes']
    graph_cols = [c for c in graph_cols if c in df_snap.columns]
    graph_metrics = (df_snap[graph_cols]
                     .groupby(['N_Bays', 'Scenario']).mean(numeric_only=True)
                     .reset_index())

    # --- Recovery Time ---
    recovery_df = compute_recovery_time(df)

    # --- CBR Top Nodes ---
    cbr_df   = compute_cbr(df)
    max_n    = df['N_Bays'].max()
    cbr_top  = (cbr_df[cbr_df['N_Bays'] == max_n]
                .sort_values('CBR', ascending=False)
                .head(30))

    try:
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            summary.rename(columns={'N_Bays': 'N Bays'}).to_excel(
                writer, sheet_name='Executive_Summary', index=False)
            graph_metrics.to_excel(writer, sheet_name='Graph_Metrics',    index=False)
            recovery_df.to_excel(writer,   sheet_name='Recovery_Time',    index=False)
            if not df_vuln.empty:
                (df_vuln.sort_values(
                    ['N_Bays', 'Scenario', 'Vulnerability_Index'],
                    ascending=[True, True, False])
                 .to_excel(writer, sheet_name='Vulnerability', index=False))
            cbr_top.to_excel(writer,      sheet_name='CBR_Top_Nodes', index=False)
            df_snap.to_excel(writer,      sheet_name='Raw_Data',      index=False)
    except Exception as exc:
        print(f"Warning -- could not write {filepath}: {exc}")


def export_metrics_by_scenario(
    df: pd.DataFrame,
    df_vuln: pd.DataFrame,
    filename: str = 'Metrics_By_Scenario.xlsx',
) -> None:
    """
    Export a clean per-scenario Excel workbook with human-readable column names.

    Sheets:
      - Scenario_Base / Scenario_Upgraded : full metric table, one row per snapshot
      - Vulnerability_Base / _Upgraded    : node vulnerability at steady state
      - CBR_Base / CBR_Upgraded           : Centrality Burst Ratio per node

    Args:
        df:       Main simulation DataFrame.
        df_vuln:  Vulnerability DataFrame.
        filename: Output filename (written to results/).
    """
    filepath = _results(filename)
    print(f"Generating per-scenario report -> {filepath}")

    df_snap = df.drop(columns=['Node_Label', 'Centrality']).drop_duplicates().copy()

    # Column selection order (sections: Identity | Topology | Latency | PLR | Switch | Compliance)
    col_order = [
        'N_Bays', 'Snapshot_Time_ms', 'Delta_T_ms', 'Total_Nodes',
        'Fiedler_Lambda2', 'Global_Efficiency', 'Load_Gini',
        'E2E_Avg_SV_us', 'E2E_Max_SV_us', 'E2E_PLR_Avg_SV', 'E2E_PLR_Max_SV', 'E2E_SV_Compliance',
        'E2E_Avg_GOOSE_us', 'E2E_Max_GOOSE_us', 'E2E_PLR_Avg_GOOSE', 'E2E_PLR_Max_GOOSE',
        'E2E_GOOSE_Compliance',
        'Latency_Avg_P6', 'Latency_Max_P6', 'Jitter_Avg_P6',
        'Loss_Avg_P6',    'Loss_Max_P6',    'Loss_P95_P6',
        'Latency_Avg_P4', 'Latency_Max_P4', 'Jitter_Avg_P4',
        'Loss_Avg_P4',    'Loss_Max_P4',    'Loss_P95_P4',
        'Latency_Avg_P5', 'Latency_Max_P5', 'Jitter_Avg_P5',
        'Loss_Avg_P5',    'Loss_Max_P5',    'Loss_P95_P5',
        'Latency_Avg_P7', 'Latency_Max_P7', 'Jitter_Avg_P7',
        'Loss_Avg_P7',    'Loss_Max_P7',    'Loss_P95_P7',
        'Max_Util_SW1', 'Max_Util_SW2',
        'IEC_Violation',
    ]
    col_order = [c for c in col_order if c in df_snap.columns]

    col_rename = {
        'N_Bays':            'N Bays',
        'Snapshot_Time_ms':  'Time Since Fault (ms)',
        'Delta_T_ms':        'GOOSE deltat (ms)',
        'Total_Nodes':       'Graph Nodes',
        'Fiedler_Lambda2':   'Algebraic Connectivity λ₂',
        'Global_Efficiency': 'Global Network Efficiency',
        'Load_Gini':         'Load Gini Coefficient',
        'E2E_Avg_SV_us':        'E2E Avg SV (us)',
        'E2E_Max_SV_us':        'E2E Max SV (us)',
        'E2E_PLR_Avg_SV':       'E2E PLR Avg SV (%)',
        'E2E_PLR_Max_SV':       'E2E PLR Max SV (%)',
        'E2E_SV_Compliance':    'SV flows < 3 ms (fraction)',
        'E2E_Avg_GOOSE_us':     'E2E Avg GOOSE (us)',
        'E2E_Max_GOOSE_us':     'E2E Max GOOSE (us)',
        'E2E_PLR_Avg_GOOSE':    'E2E PLR Avg GOOSE (%)',
        'E2E_PLR_Max_GOOSE':    'E2E PLR Max GOOSE (%)',
        'E2E_GOOSE_Compliance': 'GOOSE flows < 3 ms (fraction)',
        'Latency_Avg_P6':    'Queue wait Avg P6-Prot (ms)',
        'Latency_Max_P6':    'Queue wait Max P6-Prot (ms)',
        'Jitter_Avg_P6':     'Jitter Avg P6-Prot (ms)',
        'Loss_Avg_P6':       'PLR Avg P6-Prot (%)',
        'Loss_Max_P6':       'PLR Max P6-Prot (%)',
        'Loss_P95_P6':       'PLR P95 P6-Prot (%)',
        'Latency_Avg_P4':    'Queue wait Avg P4-SV (ms)',
        'Latency_Max_P4':    'Queue wait Max P4-SV (ms)',
        'Jitter_Avg_P4':     'Jitter Avg P4-SV (ms)',
        'Loss_Avg_P4':       'PLR Avg P4-SV (%)',
        'Loss_Max_P4':       'PLR Max P4-SV (%)',
        'Loss_P95_P4':       'PLR P95 P4-SV (%)',
        'Latency_Avg_P5':    'Queue wait Avg P5-Ctrl (ms)',
        'Latency_Max_P5':    'Queue wait Max P5-Ctrl (ms)',
        'Jitter_Avg_P5':     'Jitter Avg P5-Ctrl (ms)',
        'Loss_Avg_P5':       'PLR Avg P5-Ctrl (%)',
        'Loss_Max_P5':       'PLR Max P5-Ctrl (%)',
        'Loss_P95_P5':       'PLR P95 P5-Ctrl (%)',
        'Latency_Avg_P7':    'Queue wait Avg P7-PTP (ms)',
        'Latency_Max_P7':    'Queue wait Max P7-PTP (ms)',
        'Jitter_Avg_P7':     'Jitter Avg P7-PTP (ms)',
        'Loss_Avg_P7':       'PLR Avg P7-PTP (%)',
        'Loss_Max_P7':       'PLR Max P7-PTP (%)',
        'Loss_P95_P7':       'PLR P95 P7-PTP (%)',
        'Max_Util_SW1':      'Max Util SW1 (ratio)',
        'Max_Util_SW2':      'Max Util SW2 (ratio)',
        'IEC_Violation':     'IEC 61850 Violation (1=FAIL)',
    }

    try:
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Main data -- one sheet per scenario
            for scenario in sorted(df_snap['Scenario'].unique()):
                sheet_df = (df_snap[df_snap['Scenario'] == scenario][col_order]
                            .sort_values(['N_Bays', 'Snapshot_Time_ms'])
                            .rename(columns=col_rename)
                            .reset_index(drop=True))
                sheet_name = f'Scenario_{scenario.capitalize()}'
                sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
                ws = writer.sheets[sheet_name]
                for col_cells in ws.columns:
                    max_len = max(len(str(cell.value or '')) for cell in col_cells)
                    ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 4, 30)

            # Vulnerability -- one sheet per scenario
            if df_vuln is not None and not df_vuln.empty:
                for scenario in sorted(df_vuln['Scenario'].unique()):
                    vuln_sc = (df_vuln[df_vuln['Scenario'] == scenario]
                               .sort_values(['N_Bays', 'Vulnerability_Index'],
                                            ascending=[True, False])
                               .drop(columns=['Scenario'])
                               .reset_index(drop=True))
                    sheet_name = f'Vulnerability_{scenario.capitalize()}'
                    vuln_sc.to_excel(writer, sheet_name=sheet_name, index=False)
                    ws = writer.sheets[sheet_name]
                    for col_cells in ws.columns:
                        max_len = max(len(str(cell.value or '')) for cell in col_cells)
                        ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 4, 40)

            # CBR -- one sheet per scenario
            cbr_df = compute_cbr(df)
            if not cbr_df.empty:
                for scenario in sorted(cbr_df['Scenario'].unique()):
                    cbr_sc = (cbr_df[cbr_df['Scenario'] == scenario]
                              .sort_values(['N_Bays', 'CBR'], ascending=[True, False])
                              .drop(columns=['Scenario'])
                              .reset_index(drop=True))
                    cbr_sc.to_excel(writer, sheet_name=f'CBR_{scenario.capitalize()}', index=False)

        print(f"Per-scenario report written: {filepath}")
    except Exception as exc:
        print(f"Error writing {filepath}: {exc}")


# =============================================================================
# 3. VISUALISATION FUNCTIONS
# =============================================================================

def plot_latency_scalability(df: pd.DataFrame) -> None:
    """
    Fig 2 -- Average maximum P6 (GOOSE Protection) latency vs. number of bays.

    Shows how end-to-end protection latency grows with substation size,
    comparing base and upgraded trunk scenarios.  The IEC 61850 3 ms deadline
    is shown as a horizontal reference line.

    Args:
        df: Main simulation DataFrame.
    """
    if df.empty:
        return

    df_snap = df.drop(columns=['Node_Label', 'Centrality']).drop_duplicates().copy()
    avg_df  = df_snap.groupby(['N_Bays', 'Scenario']).mean(numeric_only=True).reset_index()
    max_bays = avg_df['N_Bays'].max()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    colors  = {'base': '#1f77b4', 'upgraded': '#2ca02c', 'bbp_only': '#ff7f0e'}
    markers = {'base': 'o',       'upgraded': 's', 'bbp_only': '^'}

    for scenario in avg_df['Scenario'].unique():
        sc = avg_df[avg_df['Scenario'] == scenario]
        ax.plot(sc['N_Bays'], sc['Latency_Max_P6'],
                color=colors.get(scenario, 'k'),
                marker=markers.get(scenario, '.'),
                linestyle='-', zorder=5,
                label=f'P6-Prot Max Avg -- {scenario}')

    ax.set_title(f'Average Max Latency vs. Number of Bays (N=1 to {max_bays})',
                 fontsize=14, weight='bold')
    ax.set_ylabel('Average Latency During Burst (ms)')
    ax.set_xlabel('Number of Bays')
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)

    plt.tight_layout()
    filepath = _results('Fig2_LatencyScalability.png')
    plt.savefig(filepath, dpi=300)
    plt.show()
    print(f"Figure saved: {filepath}")


def plot_complex_metrics(df: pd.DataFrame) -> None:
    """
    Fig 3 -- Complex network metrics vs. number of bays.

    Three panels (averaged over all burst snapshots):
      Left  : Algebraic connectivity λ₂ (Fiedler eigenvalue)
      Centre: Global network efficiency (Latora–Marchiori)
      Right : Load Gini coefficient

    Args:
        df: Main simulation DataFrame.
    """
    if df.empty:
        return

    df_snap = df.drop(columns=['Node_Label', 'Centrality']).drop_duplicates().copy()
    avg_df  = df_snap.groupby(['N_Bays', 'Scenario']).mean(numeric_only=True).reset_index()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Complex Network Metrics vs. Number of Bays', fontsize=14, weight='bold')

    colors  = {'base': '#1f77b4', 'upgraded': '#2ca02c', 'bbp_only': '#ff7f0e'}
    markers = {'base': 'o',       'upgraded': 's', 'bbp_only': '^'}

    panels = [
        ('Fiedler_Lambda2',   r'Algebraic Connectivity ($\lambda_2$)',   r'$\lambda_2$ (weighted)'),
        ('Global_Efficiency', 'Global Network Efficiency (E)', 'Efficiency (normalised)'),
        ('Load_Gini',         'Load Gini Coefficient',         'Gini Index'),
    ]

    for ax, (col, title, ylabel) in zip(axes, panels):
        if col not in avg_df.columns:
            ax.set_title(f'{title}\n(no data)')
            continue
        for scenario in avg_df['Scenario'].unique():
            sc = avg_df[avg_df['Scenario'] == scenario]
            ax.plot(sc['N_Bays'], sc[col],
                    color=colors.get(scenario, 'k'),
                    marker=markers.get(scenario, '.'),
                    linestyle='-', label=scenario)
        ax.set_title(title, fontsize=11, weight='bold')
        ax.set_xlabel('Number of Bays')
        ax.set_ylabel(ylabel)
        ax.legend(frameon=True)
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    filepath = _results('Fig3_ComplexNetworkMetrics.png')
    plt.savefig(filepath, dpi=300)
    plt.show()
    print(f"Figure saved: {filepath}")


def plot_transient_behavior(df: pd.DataFrame, n_bay_focus: int = 10) -> None:
    """
    Fig 4 -- Transient GOOSE burst latency over time for a given bay count.

    Shows how P6 (Protection) latency spikes at fault onset (t = 4 ms) and
    recovers as the GOOSE retransmission interval increases back to 1000 ms.

    Args:
        df:           Main simulation DataFrame.
        n_bay_focus:  Bay count to plot (default 10).
    """
    if df.empty:
        return

    df_snap = df.drop(columns=['Node_Label', 'Centrality']).drop_duplicates().copy()
    df_focus = df_snap[df_snap['N_Bays'] == n_bay_focus]
    if df_focus.empty:
        print(f"No data for N={n_bay_focus} -- skipping transient plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {'base': '#1f77b4', 'upgraded': '#2ca02c', 'bbp_only': '#ff7f0e'}
    for scenario in df_focus['Scenario'].unique():
        sc = df_focus[df_focus['Scenario'] == scenario]
        ax.plot(sc['Snapshot_Time_ms'], sc['Latency_Max_P6'],
                color=colors.get(scenario, 'k'),
                marker='o', linestyle='-',
                label=f'GOOSE Latency (P6) -- {scenario}')

    ax.set_title(f'Transient GOOSE Burst Analysis (N={n_bay_focus} Bays)',
                 fontsize=14, weight='bold')
    ax.set_ylabel('Maximum End-to-End Latency (ms)')
    ax.set_xlabel('Time Since Fault (ms)')
    ax.legend(loc='best', frameon=True)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)

    plt.tight_layout()
    filepath = _results(f'Fig4_GooseBurst_N{n_bay_focus}.png')
    plt.savefig(filepath, dpi=300)
    plt.show()
    print(f"Figure saved: {filepath}")


def plot_centrality_heatmap(
    df: pd.DataFrame,
    n_bay_focus: int = 10,
    scenario_focus: str = 'base',
    top_k: int = 15,
) -> None:
    """
    Fig 5 -- Betweenness centrality heatmap for the top-k critical nodes over time.

    Each row is a graph node; each column is a burst snapshot time.  High-
    centrality nodes during the burst phase indicate transient bottlenecks.

    Args:
        df:             Main simulation DataFrame.
        n_bay_focus:    Bay count to plot.
        scenario_focus: Scenario to plot ('base' or 'upgraded').
        top_k:          Number of nodes to display (sorted by peak centrality).
    """
    if df.empty:
        return

    print(f"\nGenerating centrality heatmap for N={n_bay_focus}, scenario='{scenario_focus}' ...")

    df_focus = df[(df['N_Bays'] == n_bay_focus) & (df['Scenario'] == scenario_focus)]
    if df_focus.empty:
        print("No data for this selection -- skipping heatmap.")
        return

    top_nodes = df_focus.groupby('Node_Label')['Centrality'].max().nlargest(top_k).index
    df_top    = df_focus[df_focus['Node_Label'].isin(top_nodes)]

    try:
        heatmap_data = df_top.pivot_table(
            index='Snapshot_Time_ms',
            columns='Node_Label',
            values='Centrality',
            fill_value=0,
        )
    except Exception as exc:
        print(f"Error building heatmap pivot table: {exc}")
        return

    if heatmap_data.empty:
        print("Heatmap data is empty -- skipping.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(heatmap_data.T, ax=ax, cmap='viridis', linewidths=0.5, cbar_kws={'label': 'Betweenness Centrality', 'shrink': 0.8})
    ax.set_title(
        f'Betweenness Centrality of Critical Nodes (N={n_bay_focus}, Scenario: {scenario_focus})',
        fontsize=14, weight='bold', pad=10
    )
    ax.set_xlabel('Time Since Fault (ms)')
    ax.set_ylabel('Graph Node Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    filepath = _results(f'Fig5_CentralityHeatmap_N{n_bay_focus}_{scenario_focus}.png')
    plt.savefig(filepath, dpi=300)
    plt.show()
    print(f"Figure saved: {filepath}")


def plot_cbr_vulnerability(
    df: pd.DataFrame,
    df_vuln: pd.DataFrame,
    n_bay_focus: int = 10,
    scenario_focus: str = 'base',
    top_k: int = 15,
) -> None:
    """
    Fig 6 & 7 -- Horizontal bar charts for CBR and Node Vulnerability Index.

    Left panel : Top-k nodes by Centrality Burst Ratio (CBR = C_burst / C_steady).
    Right panel: Top-k nodes by Vulnerability Index (V = deltaE / E_G).

    Args:
        df:             Main simulation DataFrame.
        df_vuln:        Vulnerability DataFrame.
        n_bay_focus:    Bay count to analyse.
        scenario_focus: Scenario to analyse.
        top_k:          Number of nodes to display in each panel.
    """
    if df.empty:
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f'Critical Node Analysis -- N={n_bay_focus} Bays, Scenario: {scenario_focus}',
        fontsize=14, weight='bold',
    )

    def shorten(lbl: str) -> str:
        return (lbl.replace('L3_VLAN_', '')
                   .replace('L4_Priority_', 'Prio_')
                   .replace('L4_Trunk_', 'Trunk_')
                   .replace('L5_OutPort_', 'Out_'))

    # Left -- CBR
    ax = axes[0]
    cbr_df    = compute_cbr(df)
    cbr_focus = (cbr_df[(cbr_df['N_Bays'] == n_bay_focus) &
                         (cbr_df['Scenario'] == scenario_focus)]
                 .nlargest(top_k, 'CBR'))

    if not cbr_focus.empty:
        short_labels = [shorten(l) for l in cbr_focus['Node_Label']]
        bars = ax.barh(short_labels[::-1], cbr_focus['CBR'].values[::-1],
                       color='#4C72B0', edgecolor='black', linewidth=0.8, alpha=0.9)
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=10)
        ax.set_xlabel('Centrality Burst Ratio (CBR)', fontsize=11)
        ax.set_title('Centrality Burst Ratio (CBR)',
                     fontsize=12, weight='bold')
        ax.legend(fontsize=10, frameon=True)
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    else:
        ax.text(0.5, 0.5, 'No CBR data', ha='center', va='center', transform=ax.transAxes)

    # Right -- Vulnerability Index
    ax = axes[1]
    if df_vuln is not None and not df_vuln.empty:
        vuln_focus = (df_vuln[(df_vuln['N_Bays'] == n_bay_focus) &
                               (df_vuln['Scenario'] == scenario_focus)]
                      .nlargest(top_k, 'Vulnerability_Index'))
        if not vuln_focus.empty:
            short_labels = [shorten(l) for l in vuln_focus['Node_Label']]
            bars = ax.barh(short_labels[::-1], vuln_focus['Vulnerability_Index'].values[::-1],
                           color='#55A868', edgecolor='black', linewidth=0.8, alpha=0.9)
            ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=10)
            ax.set_xlabel('Node Vulnerability Index ($V$)', fontsize=11)
            ax.set_title('Node Vulnerability Index',
                         fontsize=12, weight='bold')
            ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        else:
            ax.text(0.5, 0.5, 'No vulnerability data', ha='center', va='center',
                    transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'No vulnerability data', ha='center', va='center',
                transform=ax.transAxes)

    plt.tight_layout()
    filepath = _results(f'Fig67_CBR_Vulnerability_N{n_bay_focus}_{scenario_focus}.png')
    plt.savefig(filepath, dpi=300)
    plt.show()
    print(f"Figure saved: {filepath}")


def plot_sv_performance(df: pd.DataFrame) -> None:
    """
    Fig 8 -- Average End-to-End Delay and Packet Loss Rate for SV (P4) vs. bays.

    Two panels:
      Left: Average Delay (ms) for Sampled Values (P4).
      Right: Average Packet Loss Rate (%) for Sampled Values (P4).

    Args:
        df: Main simulation DataFrame.
    """
    if df.empty:
        return

    df_snap = df.drop(columns=['Node_Label', 'Centrality']).drop_duplicates().copy()
    
    # Group by scenario and N_bays to find the maximum over all burst snapshots
    # (worst-case moment) of the *average* delay and *average* PLR for P4 traffic.
    worst = (df_snap
             .groupby(['N_Bays', 'Scenario'])[['E2E_Avg_SV_us', 'E2E_PLR_Avg_SV']]
             .max()
             .reset_index())

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle('Sampled Values (SV - P4) Performance vs. Number of Bays',
                 fontsize=14, weight='bold')

    scenario_style = {
        'base':     {'linestyle': '-',  'marker': 'o', 'label': 'Base (100 Mbps)'},
        'upgraded': {'linestyle': '--', 'marker': 's', 'label': 'Trunk + BBP link 1 Gbps'},
        'bbp_only': {'linestyle': '-.', 'marker': '^', 'label': 'BBP link only 1 Gbps'},
    }
    scenario_color = {'base': '#1f77b4', 'upgraded': '#2ca02c', 'bbp_only': '#ff7f0e'}

    # 1. Left Panel: Average Delay
    ax = axes[0]
    for scenario in worst['Scenario'].unique():
        sc = worst[worst['Scenario'] == scenario].sort_values('N_Bays')
        style = scenario_style.get(scenario, {'linestyle': '-', 'marker': '.', 'label': scenario})
        ax.plot(sc['N_Bays'], sc['E2E_Avg_SV_us'],
                color=scenario_color.get(scenario, 'k'),
                linewidth=2.0, alpha=0.9 if scenario == 'base' else 0.8,
                zorder=5, **style)
    ax.axhline(T_MAX * 1e6, color='red', linestyle=':', linewidth=1.5,
               label='IEC 61850 limit (3 ms)')
    ax.set_yscale('log')
    ax.set_title('Average End-to-End Delay (SV)', fontsize=12, weight='bold')
    ax.set_xlabel('Number of Bays (N)')
    ax.set_ylabel('Average Delay (µs)')
    ax.set_xticks(sorted(worst['N_Bays'].unique()))
    ax.legend(fontsize=10, frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)

    # 2. Right Panel: Average PLR
    ax = axes[1]
    for scenario in worst['Scenario'].unique():
        sc = worst[worst['Scenario'] == scenario].sort_values('N_Bays')
        style = scenario_style.get(scenario, {'linestyle': '-', 'marker': '.', 'label': scenario})
        ax.plot(sc['N_Bays'], sc['E2E_PLR_Avg_SV'],
                color=scenario_color.get(scenario, 'k'),
                linewidth=2.0, alpha=0.9 if scenario == 'base' else 0.8,
                zorder=5, **style)
    ax.set_title('Average Packet Loss Rate (SV)', fontsize=12, weight='bold')
    ax.set_xlabel('Number of Bays (N)')
    ax.set_ylabel('Average PLR (%)')
    ax.set_xticks(sorted(worst['N_Bays'].unique()))
    ax.legend(fontsize=10, frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    filepath = _results('Fig8_SV_Performance.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Figure saved: {filepath}")

# ===============================================================================================
# 4. NETWORK TOPOLOGY VISUALISATION
# =============================================================================

def _hierarchical_pos(labels: list[str]) -> dict[int, tuple[float, float]]:
    """
    Assign (x, y) coordinates for the 5-layer hierarchical layout.

    Y-bands (bottom to top):
      0 -> L5 OutPort
      1 -> L4 Priority / Trunk
      2 -> L3 VLAN
      3 -> L2 InPort
      4 -> L1 Equipment

    Within each band, nodes are evenly spaced along the x-axis.
    For L3, SW1 nodes occupy the left half and SW2 nodes the right half,
    mirroring the physical topology split (N > 5).
    """
    # Bucket each node index into its layer
    buckets: dict[int, list[int]] = {k: [] for k in range(5)}

    for idx, lbl in enumerate(labels):
        if 'L1_Equipment' in lbl:
            buckets[4].append(idx)
        elif 'L2_InPort' in lbl:
            buckets[3].append(idx)
        elif 'L3_VLAN' in lbl:
            buckets[2].append(idx)
        elif 'L4_' in lbl:
            buckets[1].append(idx)
        elif 'L5_OutPort' in lbl:
            buckets[0].append(idx)

    # For L3, separate SW1 (left) and SW2 (right) to reflect physical split
    l3_sw1 = [i for i in buckets[2] if labels[i].endswith('_SW1')]
    l3_sw2 = [i for i in buckets[2] if labels[i].endswith('_SW2')]
    l3_ordered = l3_sw1 + l3_sw2
    buckets[2] = l3_ordered  # SW1 left, SW2 right

    pos: dict[int, tuple[float, float]] = {}
    for y_band, node_list in buckets.items():
        n = len(node_list)
        for k, node_idx in enumerate(node_list):
            x = (k + 0.5) / max(n, 1)
            pos[node_idx] = (x, float(y_band))

    return pos


def _get_node_style(label: str) -> tuple[str, float]:
    """Return (hex_color, relative_size_factor) for a node given its label."""
    if 'L1_Equipment' in label:
        return '#4C72B0', 1.0   # Muted Blue
    if 'L2_InPort'    in label:
        return '#55A868', 0.8   # Muted Green
    if 'L3_VLAN'      in label:
        return '#F5961D', 0.5   # Muted Orange
    if 'L4_Trunk'     in label:
        return '#8172B3', 1.6   # Muted Purple
    if 'L4_Priority'  in label:
        return '#C44E52', 1.2   # Muted Red
    if 'L5_OutPort'   in label:
        return '#64B5CD', 0.8   # Cyan
    return '#8C8C8C', 0.7


def _shorten_label(label: str) -> str:
    """Strip layer prefix for a compact but readable node label."""
    for prefix in ('L1_Equipment_', 'L2_InPort_', 'L3_VLAN_',
                   'L4_Priority_', 'L4_Trunk_', 'L5_OutPort_'):
        if label.startswith(prefix):
            return label[len(prefix):]
    return label


def plot_network_graphs(max_bays: int = 10, scenario: str = 'base') -> None:
    """
    Generate one publication-quality network topology figure per bay count.

    Each figure shows the 5-layer directed graph with a hierarchical band
    layout (L1 at top, L5 at bottom).  Edges are coloured by their
    normalised weight (heavier = higher latency).  L5->L1 feedback edges
    are omitted to keep the figure uncluttered.

    Figures are saved to  results/graphs/  at 300 dpi.

    Args:
        max_bays: Generate figures for N = 1 … max_bays (default 10).
        scenario: Traffic scenario used for edge-weight colouring ('base').
    """
    graphs_dir = os.path.join(RESULTS_DIR, 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)

    # Layer-band y-labels (bottom to top)
    band_labels = {
        0: 'L5  OutPort',
        1: 'L4  Priority / Trunk',
        2: 'L3  VLAN @ Switch',
        3: 'L2  InPort',
        4: 'L1  Equipment',
    }
    # Legend entries: (color, label)
    layer_legend = [
        ('#2196F3', 'L1 Equipment'),
        ('#4CAF50', 'L2 InPort'),
        ('#FF9800', 'L3 VLAN'),
        ('#F44336', 'L4 Priority'),
        ('#B71C1C', 'L4 Trunk'),
        ('#9C27B0', 'L5 OutPort'),
    ]

    for n_bays in range(1, max_bays + 1):
        print(f"  Generating network graph N={n_bays} ...")

        # Build topology and compute steady-state weights
        devices, sw_map = generate_devices(n_bays)
        D_base          = generate_demand_tensor(n_bays, devices, delta_t=1000.0)
        labels, A       = build_topology(n_bays, devices, D_base, sw_map)
        W, _J, _PL, _u  = compute_weights(labels, A, D_base, sw_map, devices,
                                           1000.0, scenario)

        # Build directed graph (forward edges only -- skip L5->L1 feedback)
        G = nx.DiGraph()
        for idx in range(len(labels)):
            G.add_node(idx)

        w_max = W.max() if W.max() > 0 else 1.0
        for i in range(len(labels)):
            for j in range(len(labels)):
                if W[i, j] > 0:
                    if 'L5_OutPort' in labels[i] and 'L1_Equipment' in labels[j]:
                        continue   # Omit feedback cycle
                    G.add_edge(i, j, weight=W[i, j])

        pos = _hierarchical_pos(labels)

        # ── Figure layout ─────────────────────────────────────────────────────
        fig_w, fig_h = max(12.0, 7.0 + n_bays * 0.8), 9.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        # Horizontal subtle dashed grids for layers
        for y_band in range(5):
            ax.axhline(y_band, color='#E0E0E0', linestyle='--', linewidth=1.2, zorder=0)

        if G.number_of_edges() > 0:
            edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
            edge_widths  = [0.5 + 2.5 * (w / w_max) for w in edge_weights]

            nx.draw_networkx_edges(
                G, pos, ax=ax,
                width=edge_widths,
                edge_color=edge_weights,
                edge_cmap=plt.cm.coolwarm,
                edge_vmin=0, edge_vmax=w_max,
                alpha=0.65,
                arrows=True,
                arrowsize=max(5, 12 - n_bays),
                arrowstyle='-|>',
                connectionstyle='arc3,rad=0.08',
                min_source_margin=5,
                min_target_margin=5,
            )

        # ── Nodes ─────────────────────────────────────────────────────────────
        base_size = max(15, 200 - n_bays * 18)
        for idx in G.nodes():
            lbl   = labels[idx]
            color, size_factor = _get_node_style(lbl)
            x, y  = pos[idx]
            nodes = nx.draw_networkx_nodes(
                G, pos, nodelist=[idx], ax=ax,
                node_color=color,
                edgecolors='black',
                linewidths=1.0,
                node_size=base_size * size_factor,
                alpha=1.0,
            )
            nodes.set_zorder(2)

        # ── Node labels ───────────────────────────────────────────────────────
        show_all  = n_bays <= 3
        show_some = 4 <= n_bays <= 6   # Only global devices

        if show_all or show_some:
            label_dict = {}
            for idx in G.nodes():
                lbl = labels[idx]
                if show_all:
                    label_dict[idx] = _shorten_label(lbl)
                else:
                    # Only show GPS, BBP, SMC, trunk nodes
                    short = _shorten_label(lbl)
                    if any(k in lbl for k in ('GPS', 'BBP', 'SMC', 'Trunk')):
                        label_dict[idx] = short

            if label_dict:
                font_size = max(5, 9 - n_bays)
                nx.draw_networkx_labels(
                    G, pos, labels=label_dict, ax=ax,
                    font_size=font_size, font_family='sans-serif',
                    font_color='black', font_weight='bold',
                )

        # ── Band annotations (right margin) ───────────────────────────────────
        for y_band, band_text in band_labels.items():
            ax.text(1.01, y_band, band_text,
                    transform=ax.get_yaxis_transform(),
                    fontsize=9, color='#555555',
                    va='center', ha='left', style='italic')

        # SW split annotation for N > 5
        if n_bays > 5:
            ax.text(0.25, 2, '◀ SW1', fontsize=8, color='#555555',
                    ha='center', va='bottom', transform=ax.transData)
            ax.text(0.75, 2, 'SW2 ▶', fontsize=8, color='#555555',
                    ha='center', va='bottom', transform=ax.transData)

        # ── Legend ────────────────────────────────────────────────────────────
        from matplotlib.patches import Patch
        # Sync with _get_node_style
        layer_legend = [
            ('#4C72B0', 'L1 Equipment'),
            ('#55A868', 'L2 InPort'),
            ('#F5961D', 'L3 VLAN'),
            ('#C44E52', 'L4 Priority'),
            ('#8172B3', 'L4 Trunk'),
            ('#64B5CD', 'L5 OutPort'),
        ]
        legend_handles = [
            Patch(facecolor=c, edgecolor='black', linewidth=1.0, label=lbl)
            for c, lbl in layer_legend
        ]
        ax.legend(handles=legend_handles, loc='upper left',
                  fontsize=9, framealpha=1.0, edgecolor='black', title='Graph Layer',
                  title_fontsize=10)

        # ── Titles and axes ───────────────────────────────────────────────────
        n_nodes = len(labels)
        n_edges = G.number_of_edges()
        ax.set_title(
            f'IEC 61850 Substation -- 5-Layer Complex Network\n'
            f'N = {n_bays} Bay{"s" if n_bays > 1 else ""}  '
            f'({n_nodes} nodes, {n_edges} edges)  |  Scenario: {scenario}',
            fontsize=12, weight='bold', pad=10,
        )
        ax.set_yticks(list(band_labels.keys()))
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.6, 4.6)
        ax.spines[:].set_visible(False)

        plt.tight_layout()
        filepath = os.path.join(graphs_dir, f'Network_{scenario.capitalize()}_N{n_bays:02d}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        # Also save N=1 base as the paper's Figure 1 in the main results folder
        if n_bays == 1 and scenario == 'base':
            fig1_path = _results('Fig1_NetworkModel_Base_N1.png')
            plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
            print(f"    Saved: {fig1_path}")
        plt.close(fig)   # Don't display interactively -- 10 figures would flood the screen
        print(f"    Saved: {filepath}")

    print(f"\nAll network graphs saved to: {graphs_dir}/")


def plot_fault_evolution_snapshots(n_bays: int = 1, scenario: str = 'base') -> None:
    """
    Fig 9 -- Graph weight evolution during the GOOSE burst fault event.
    Generates a 2x2 subplot figure with 1 pre-fault snapshot and 3 post-fault snapshots.
    Links are coloured according to their current weight.
    """
    print(f"\nGenerating fault evolution snapshots (N={n_bays}, scenario='{scenario}') ...")
    
    t_snapshots, delta_t_list = generate_goose_burst_timeline()
    
    devices, sw_map = generate_devices(n_bays)
    D_base = generate_demand_tensor(n_bays, devices, delta_t=1000.0)
    labels, A = build_topology(n_bays, devices, D_base, sw_map)
    
    indices = [0, 1, 2, 3] # Pre-fault, and 1st, 2nd, 3rd post-fault
    snapshot_data = []
    
    global_w_max = 0.0
    
    for idx in indices:
        t = t_snapshots[idx]
        dt = delta_t_list[idx]
        
        D = update_goose_demand(D_base, n_bays, dt)
        W, _J, _PL, _u = compute_weights(labels, A, D, sw_map, devices, dt, scenario)
        
        if W.max() > global_w_max:
            global_w_max = W.max()
            
        snapshot_data.append((t, dt, W))
        
    if global_w_max == 0:
        global_w_max = 1.0
        
    pos = _hierarchical_pos(labels)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Graph Weight Evolution During GOOSE Burst Fault\n(N={n_bays} Bay{"s" if n_bays > 1 else ""}, Scenario: {scenario})',
                 fontsize=16, weight='bold')
    axes_flat = axes.flatten()
    
    band_labels = {
        0: 'L5 OutPort',
        1: 'L4 Priority/Trunk',
        2: 'L3 VLAN @ Switch',
        3: 'L2 InPort',
        4: 'L1 Equipment',
    }
    
    for ax, (t, dt, W) in zip(axes_flat, snapshot_data):
        ax.set_facecolor('white')
        
        for y_band in range(5):
            ax.axhline(y_band, color='#E0E0E0', linestyle='--', linewidth=1.0, zorder=0)
            
        G = nx.DiGraph()
        for i in range(len(labels)):
            G.add_node(i)
            
        for i in range(len(labels)):
            for j in range(len(labels)):
                if W[i, j] > 0:
                    if 'L5_OutPort' in labels[i] and 'L1_Equipment' in labels[j]:
                        continue
                    G.add_edge(i, j, weight=W[i, j])
                    
        if G.number_of_edges() > 0:
            edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
            edge_widths = [0.5 + 2.5 * (w / global_w_max) for w in edge_weights]
            
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                width=edge_widths,
                edge_color=edge_weights,
                edge_cmap=plt.cm.coolwarm,
                edge_vmin=0, edge_vmax=global_w_max,
                alpha=0.75,
                arrows=True,
                arrowsize=max(5, 12 - n_bays),
                arrowstyle='-|>',
                connectionstyle='arc3,rad=0.08',
                min_source_margin=5,
                min_target_margin=5,
            )
            
        base_size = max(15, 200 - n_bays * 18)
        for idx in G.nodes():
            lbl = labels[idx]
            color, size_factor = _get_node_style(lbl)
            nodes = nx.draw_networkx_nodes(
                G, pos, nodelist=[idx], ax=ax,
                node_color=color, edgecolors='black',
                linewidths=1.0, node_size=base_size * size_factor,
                alpha=1.0,
            )
            nodes.set_zorder(2)
            
        show_all = n_bays <= 3
        show_some = 4 <= n_bays <= 6
        if show_all or show_some:
            label_dict = {}
            for idx in G.nodes():
                lbl = labels[idx]
                if show_all:
                    label_dict[idx] = _shorten_label(lbl)
                else:
                    short = _shorten_label(lbl)
                    if any(k in lbl for k in ('GPS', 'BBP', 'SMC', 'Trunk')):
                        label_dict[idx] = short
            if label_dict:
                font_size = max(5, 9 - n_bays)
                nx.draw_networkx_labels(
                    G, pos, labels=label_dict, ax=ax,
                    font_size=font_size, font_family='sans-serif',
                    font_color='black', font_weight='bold',
                )
            
        if t == 0:
            status = "Pre-Fault Steady State"
        else:
            status = f"Post-Fault (Δt = {dt} ms)"
            
        ax.set_title(f't = {t} ms [{status}]', fontsize=12, weight='bold')
        
        ax.set_yticks(list(band_labels.keys()))
        ax.set_yticklabels(list(band_labels.values()), fontsize=8)
        ax.set_xticks([])
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.6, 4.6)
        ax.spines[:].set_visible(False)
        ax.grid(False)
    
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=0, vmax=global_w_max))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Normalised Edge Weight (Proxy for Latency)', fontsize=12, weight='bold')
    
    plt.subplots_adjust(left=0.08, right=0.9, wspace=0.15, hspace=0.25)
    
    filepath = _results(f'Fig9_Evolution_Snapshots_N{n_bays}_{scenario}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {filepath}")

# =============================================================================
# 5. ENTRY POINT
# =============================================================================

if __name__ == '__main__':

    scenarios_to_run = ['base', 'upgraded', 'bbp_only']
    all_results = []
    all_vuln    = []

    for scenario in scenarios_to_run:
        df_scenario, df_vuln_scenario = run_simulation(max_bays=10, scenario=scenario)
        all_results.append(df_scenario)
        all_vuln.append(df_vuln_scenario)

    df_results  = pd.concat(all_results, ignore_index=True)
    df_vuln_all = pd.concat(all_vuln,    ignore_index=True)

    print('\nFinal results (first rows):')
    print(df_results.head())

    # --- Excel exports ---
    export_organized_report(df_results, df_vuln_all, 'Global_Report.xlsx')
    export_metrics_by_scenario(df_results, df_vuln_all, 'Metrics_By_Scenario.xlsx')

    # --- Figures for the journal paper ---
    # Fig 2: Latency scalability vs N
    plot_latency_scalability(df_results)

    # Fig 3: Complex network metrics (λ₂, Efficiency, Gini) vs N
    plot_complex_metrics(df_results)

    # Fig 4: GOOSE burst transient for N=10
    plot_transient_behavior(df_results, n_bay_focus=10)

    # Fig 5: Betweenness centrality heatmap (N=10, base scenario)
    plot_centrality_heatmap(df_results, n_bay_focus=10, scenario_focus='base')

    # Fig 6 & 7: CBR + Vulnerability for N=10 (both scenarios)
    plot_cbr_vulnerability(df_results, df_vuln_all, n_bay_focus=10, scenario_focus='base')
    plot_cbr_vulnerability(df_results, df_vuln_all, n_bay_focus=10, scenario_focus='upgraded')

    # Fig 8: Consolidated SV Performance (Delay & PLR for P4)
    plot_sv_performance(df_results)

    # Fig 1 + Network topology figures (base scenario, N=1..10)
    # N=1 base is also saved as Fig1_NetworkModel_Base_N1.png
    plot_network_graphs(max_bays=10, scenario='base')
    
    # Fig 9: Evolution of the graph during GOOSE Burst Fault (base scenario, N=1)
    plot_fault_evolution_snapshots(n_bays=1, scenario='base')
