# -*- coding: utf-8 -*-
"""
Simulación de Escalabilidad de Subestación IEC 61850 (Gemelo Digital)
Refactorizado y Optimizado.

Este script simula el comportamiento de una red de subestación al incrementar
el número de bahías (N), calculando métricas de latencia, jitter, pérdida de paquetes
y robustez topológica.
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import scipy.linalg as la

# =================================================================
# 1. CONFIGURACIÓN GLOBAL Y CONSTANTES
# =================================================================

# Parámetros de Tiempo y Capacidad
T_MAX = 0.003           # 3 ms (Límite IEC 61850 para SV/GOOSE críticos)
CAPACITY_DEFAULT = 100e6 # 100 Mbps
CAPACITY_TRUNK = 100e6   # 100 Mbps
FABRIC_CAPACITY = 68000.0 # Mbps (Backplane del Switch)
T_PROC_SWITCH = 4e-6     # 4 us (Latencia de conmutación)
K_BUFFER = np.floor(4e6 / 150) # Tamaño del buffer (aprox en paquetes)
FACTOR_16 = 16          # Factor de escala para fórmula de cola

# Mapa de Frecuencias (Hz) y Tamaños (Bytes) por Protocolo
TRAFFIC_SPECS = {
    "PTP":   {"freq": 3,    "size": 80},
    "SV":    {"freq": 4800, "size": 149},
    "GOOSE": {"freq": 1,    "size": 187}, # Promedio/Keep-alive
    "MON":   {"freq": 1,    "size": 171}, # Monitoreo general
    "SMC":   {"freq": 1,    "size": 211}, # SMC Request
    "MU_RES":{"freq": 1,    "size": 340}  # MU Response
}

# Mapeo de VLAN a Prioridad (IEEE 802.1Q / IEC 61850)
def get_vlan_priority(vlan_name):
    if "PTP" in vlan_name: return 7
    if "V5" in vlan_name: return 6 # BBP Global
    
    # Extraer ID de VLAN (ej: ..._V12 -> 12)
    try:
        vid = int(vlan_name.split('_V')[-1])
        if vid in [1, 2, 14, 15, 16, 17]: return 4 # SV / Low Prio
        if vid in [6, 7, 11, 13]:         return 5 # Control / Estado
        if vid in [3, 4, 8, 9, 10, 12]:   return 6 # Protección Crítica
    except:
        pass
    return 4 # Default

def get_mbps(size_bytes, freq_hz):
    """Calcula el ancho de banda en Mbps."""
    return (size_bytes * 8 * freq_hz) / 1e6

# =================================================================
# 2. GENERACIÓN DE MODELO (DISPOSITIVOS Y DEMANDA)
# =================================================================

def generar_dispositivos(n_bahias):
    """Genera la lista de dispositivos y su mapa de switches."""
    devices_global = ['GPS', 'BBP', 'SMC']
    devices_bay = []
    for n in range(1, n_bahias + 1):
        devices_bay.extend([f'PP1B{n}L', f'PP2B{n}L', f'MU1B{n}', f'MU2B{n}'])
    
    all_devices = devices_global + devices_bay
    
    # Mapa de ubicación física (Balanceo de carga si N > 5)
    sw_map = {d: "SW1" for d in devices_global}
    for n in range(1, n_bahias + 1):
        target_sw = "SW1" if (n_bahias <= 5 or n % 2 != 0) else "SW2"
        for unit in [f'PP1B{n}L', f'PP2B{n}L', f'MU1B{n}', f'MU2B{n}']:
            sw_map[unit] = target_sw
            
    return all_devices, sw_map

def generar_tensor_demanda(n_bahias, all_devices):
    """
    Crea el Tensor de Demanda D (Diccionario de DataFrames).
    Define quién envía cuánto tráfico a quién.
    """
    # Definición de claves de VLAN
    vlan_ids_bay = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    vlan_keys = ["L3_VLAN_PTP", "L3_VLAN_V5"]
    for n in range(1, n_bahias + 1):
        for vid in vlan_ids_bay:
            vlan_keys.append(f"L3_VLAN_B{n}_V{vid}")

    # Inicializar Tensor
    D = {vlan: pd.DataFrame(0.0, index=all_devices, columns=all_devices) for vlan in vlan_keys}

    # --- A. Tráfico Global ---
    bw_ptp = get_mbps(TRAFFIC_SPECS["PTP"]["size"], TRAFFIC_SPECS["PTP"]["freq"])
    for target in all_devices:
        if target != 'GPS': D["L3_VLAN_PTP"].at['GPS', target] = bw_ptp

    bw_v5 = get_mbps(179, 1)
    for n in range(1, n_bahias + 1):
        targets = [f'PP1B{n}L', f'PP2B{n}L', f'MU1B{n}', f'MU2B{n}', 'SMC']
        for t in targets: D["L3_VLAN_V5"].at['BBP', t] = bw_v5

    # --- B. Tráfico por Bahía ---
    for n in range(1, n_bahias + 1):
        pp1, pp2 = f"PP1B{n}L", f"PP2B{n}L"
        mu1, mu2 = f"MU1B{n}", f"MU2B{n}"
        
        # SV (Sampled Values)
        bw_sv = get_mbps(TRAFFIC_SPECS["SV"]["size"], TRAFFIC_SPECS["SV"]["freq"])
        D[f"L3_VLAN_B{n}_V1"].at[mu1, 'BBP'] = bw_sv
        D[f"L3_VLAN_B{n}_V1"].at[mu1, pp1] = bw_sv
        D[f"L3_VLAN_B{n}_V2"].at[mu2, 'BBP'] = bw_sv
        D[f"L3_VLAN_B{n}_V2"].at[mu2, pp2] = bw_sv
        D[f"L3_VLAN_B{n}_V16"].at[mu1, pp1] = bw_sv
        D[f"L3_VLAN_B{n}_V17"].at[mu2, pp2] = bw_sv

        # GOOSE Protección
        bw_goose = get_mbps(TRAFFIC_SPECS["GOOSE"]["size"], TRAFFIC_SPECS["GOOSE"]["freq"])
        D[f"L3_VLAN_B{n}_V3"].at[pp1, pp2] = bw_goose
        D[f"L3_VLAN_B{n}_V3"].at[pp1, mu1] = bw_goose
        D[f"L3_VLAN_B{n}_V4"].at[pp2, pp1] = bw_goose
        D[f"L3_VLAN_B{n}_V4"].at[pp2, mu2] = bw_goose

        # Control/Estado (SMC <-> MU)
        D[f"L3_VLAN_B{n}_V6"].at['SMC', mu1] = get_mbps(TRAFFIC_SPECS["SMC"]["size"], 1)
        D[f"L3_VLAN_B{n}_V6"].at[mu1, 'SMC'] = get_mbps(TRAFFIC_SPECS["MU_RES"]["size"], 1)
        D[f"L3_VLAN_B{n}_V7"].at['SMC', mu2] = get_mbps(TRAFFIC_SPECS["SMC"]["size"], 1)
        D[f"L3_VLAN_B{n}_V7"].at[mu2, 'SMC'] = get_mbps(TRAFFIC_SPECS["MU_RES"]["size"], 1)

        # Monitoreo (V8-V15)
        # Simplificación: Usamos specs de MON para todos estos flujos
        bw_mon = get_mbps(TRAFFIC_SPECS["MON"]["size"], 1)
        for v in range(8, 16):
            vlan = f"L3_VLAN_B{n}_V{v}"
            # Asignación genérica basada en lógica original
            if v in [8, 10, 11, 14]: src = mu1
            else: src = mu2
            
            if v in [8, 9]: dst = 'BBP'
            elif v in [14, 15]: dst = 'SMC'
            elif v in [10, 11]: dst = pp1
            else: dst = pp2
            
            D[vlan].at[src, dst] = bw_mon

    return D

# =================================================================
# 3. CONSTRUCCIÓN DE TOPOLOGÍA (GRAFO)
# =================================================================

def construir_topologia(n_bahias, all_devices, D, sw_map):
    """
    Construye la matriz de adyacencia A y la lista de etiquetas de nodos.
    Capas: L1(Dev) -> L2(InPort) -> L3(VLAN) -> L4(Prio/Trunk) -> L5(OutPort) -> L1
    """
    # --- Etiquetas Base ---
    labels_l1 = [f"L1_Equipment_{d}" for d in all_devices]
    labels_l2 = [f"L2_InPort_{d}" for d in all_devices]
    
    # --- Etiquetas L3 (VLANs localizadas en Switches) ---
    vlan_presence = {vlan: set() for vlan in D.keys()}
    for vlan, df in D.items():
        # Dispositivos activos en esta VLAN (fila o columna > 0)
        active_devs = df.index[df.any(axis=1)].union(df.columns[df.any(axis=0)])
        for d in active_devs:
            vlan_presence[vlan].add(sw_map[d])
            
    labels_l3 = []
    for vlan, switches in vlan_presence.items():
        for sw in sorted(list(switches)):
            labels_l3.append(f"{vlan}_{sw}")

    # --- Etiquetas L4 (Prioridades y Trunks) ---
    prio_presence = set()
    for l3 in labels_l3:
        # l3 formato: L3_VLAN_..._SWx
        parts = l3.rsplit('_', 1)
        vlan_base, sw_id = parts[0], parts[1]
        prio = get_vlan_priority(vlan_base)
        prio_presence.add((sw_id, prio))
        
    labels_l4 = [f"L4_Priority_{sw}_Prio_{p}" for sw, p in sorted(list(prio_presence))]
    
    if n_bahias > 5:
        labels_l4.extend(["L4_Trunk_SW1_to_SW2", "L4_Trunk_SW2_to_SW1"])

    # --- Etiquetas L5 (Puertos Salida) ---
    labels_l5 = [f"L5_OutPort_{d}" for d in all_devices if d != 'GPS']

    # --- Consolidación ---
    labels = labels_l1 + labels_l2 + labels_l3 + labels_l4 + labels_l5
    n_nodes = len(labels)
    A = np.zeros((n_nodes, n_nodes), dtype=int)
    
    # Helper para conectar por nombre
    def add_edge(u, v):
        try:
            A[labels.index(u), labels.index(v)] = 1
        except ValueError:
            pass # Ignorar si el nodo no existe (ej. GPS en L5)

    # --- CONEXIONES ---
    
    # 1. L1 -> L2
    for d in all_devices:
        add_edge(f"L1_Equipment_{d}", f"L2_InPort_{d}")

    # 2. L2 -> L3 (Basado en demanda)
    for vlan, df in D.items():
        senders = df.index[df.any(axis=1)]
        for src in senders:
            sw = sw_map[src]
            add_edge(f"L2_InPort_{src}", f"{vlan}_{sw}")

    # 3. L3 -> L4 (VLAN a Prioridad Local)
    for l3 in labels_l3:
        parts = l3.rsplit('_', 1)
        vlan_base, sw = parts[0], parts[1]
        prio = get_vlan_priority(vlan_base)
        add_edge(l3, f"L4_Priority_{sw}_Prio_{prio}")

    # 4. L4 -> Trunks (Inter-switch)
    if n_bahias > 5:
        # Conexión física Trunk
        add_edge("L4_Trunk_SW1_to_SW2", "L4_Trunk_SW2_to_SW1")
        add_edge("L4_Trunk_SW2_to_SW1", "L4_Trunk_SW1_to_SW2")
        
        # Lógica de enrutamiento
        for vlan, df in D.items():
            prio = get_vlan_priority(vlan)
            # Si hay flujo SW1 -> SW2
            if any((sw_map[s] == "SW1" and sw_map[d] == "SW2" and df.at[s,d]>0) for s in df.index for d in df.columns):
                add_edge(f"L4_Priority_SW1_Prio_{prio}", "L4_Trunk_SW1_to_SW2")
                add_edge("L4_Trunk_SW2_to_SW1", f"{vlan}_SW2") # Salida del trunk a VLAN destino
            
            # Si hay flujo SW2 -> SW1
            if any((sw_map[s] == "SW2" and sw_map[d] == "SW1" and df.at[s,d]>0) for s in df.index for d in df.columns):
                add_edge(f"L4_Priority_SW2_Prio_{prio}", "L4_Trunk_SW2_to_SW1")
                add_edge("L4_Trunk_SW1_to_SW2", f"{vlan}_SW1")

    # 5. L4 -> L5 (Prioridad a Puerto Salida)
    for vlan, df in D.items():
        prio = get_vlan_priority(vlan)
        for src in df.index:
            for dst in df.columns:
                if df.at[src, dst] > 0:
                    sw_dst = sw_map[dst]
                    add_edge(f"L4_Priority_{sw_dst}_Prio_{prio}", f"L5_OutPort_{dst}")

    # 6. L5 -> L1 (Cierre de bucle)
    for d in all_devices:
        if d != 'GPS':
            add_edge(f"L5_OutPort_{d}", f"L1_Equipment_{d}")

    return labels, A

# =================================================================
# 4. CÁLCULO DE PESOS (LATENCIA, JITTER, PÉRDIDA)
# =================================================================

def calcular_pesos(n_bahias, labels, A, D, sw_map, all_devices):
    """
    Calcula las matrices de Peso (W), Jitter (J) y Pérdida (PL).
    Implementa modelos M/M/1 y M/M/1/K.
    """
    n_nodes = len(labels)
    W = np.zeros((n_nodes, n_nodes), dtype=float)
    J = np.zeros((n_nodes, n_nodes), dtype=float)
    PL = np.zeros((n_nodes, n_nodes), dtype=float)
    
    # --- 1. L1 -> L2 (Transmisión de Entrada) ---
    for d in all_devices:
        sum_size_freq = 0.0
        sum_freq = 0.0
        
        # Calcular S_avg ponderado
        for vlan, df in D.items():
            if df.loc[d].any():
                # Determinar specs
                specs = TRAFFIC_SPECS["MON"] # Default
                if "PTP" in vlan: specs = TRAFFIC_SPECS["PTP"]
                elif "V5" in vlan: specs = {"freq": 1, "size": 179}
                elif "SV" in vlan or "_V1" in vlan or "_V2" in vlan: specs = TRAFFIC_SPECS["SV"]
                elif "GOOSE" in vlan or "_V3" in vlan: specs = TRAFFIC_SPECS["GOOSE"]
                elif "_V6" in vlan or "_V7" in vlan:
                    specs = TRAFFIC_SPECS["SMC"] if d == 'SMC' else TRAFFIC_SPECS["MU_RES"]
                
                sum_size_freq += (specs["size"] * specs["freq"])
                sum_freq += specs["freq"]
        
        if sum_freq > 0:
            s_avg = sum_size_freq / sum_freq
            peso = ((s_avg * 8) / CAPACITY_DEFAULT) / T_MAX
            
            try:
                idx_src = labels.index(f"L1_Equipment_{d}")
                idx_dst = labels.index(f"L2_InPort_{d}")
                W[idx_src, idx_dst] = peso
            except ValueError: pass

    # --- 2. L2 -> L3 (Procesamiento Fijo) ---
    peso_proc = T_PROC_SWITCH / T_MAX
    for i, label in enumerate(labels):
        if "L2_InPort" in label:
            for j in np.where(A[i, :] == 1)[0]:
                if "L3_VLAN" in labels[j]:
                    W[i, j] = peso_proc

    # --- 3. L3 -> L4 (Fabric Utilization - HOL Blocking) ---
    # Calcular carga por switch y prioridad
    lambda_sw = {"SW1": {p: 0.0 for p in range(8)}, "SW2": {p: 0.0 for p in range(8)}}
    
    for vlan, df in D.items():
        prio = get_vlan_priority(vlan)
        for src in df.index:
            destinos = df.columns[df.loc[src] > 0].tolist()
            if not destinos: continue
            
            bw = df.loc[src, destinos[0]]
            sw_s = sw_map[src]
            
            # Conteo de destinos
            dests_sw1 = sum(1 for d in destinos if sw_map[d] == "SW1")
            dests_sw2 = sum(1 for d in destinos if sw_map[d] == "SW2")
            
            # Carga en SW Origen
            replicas = dests_sw1 if sw_s == "SW1" else dests_sw2
            if (sw_s == "SW1" and dests_sw2 > 0) or (sw_s == "SW2" and dests_sw1 > 0):
                replicas += 1 # Copia al Trunk
            
            if sw_s in lambda_sw: lambda_sw[sw_s][prio] += (bw * replicas)
            
            # Carga en SW Destino (vía Trunk)
            sw_remoto = "SW2" if sw_s == "SW1" else "SW1"
            dests_remotos = dests_sw2 if sw_s == "SW1" else dests_sw1
            if dests_remotos > 0 and sw_remoto in lambda_sw:
                lambda_sw[sw_remoto][prio] += (bw * dests_remotos)

    # Aplicar pesos HOL
    u_p = {sw: {} for sw in ["SW1", "SW2"]}
    for sw in ["SW1", "SW2"]:
        if sw not in lambda_sw: continue
        for p in range(7, -1, -1):
            # Suma acumulativa de prioridades superiores
            load_accum = sum(lambda_sw[sw][k] for k in range(p, 8))
            u_val = load_accum / FABRIC_CAPACITY
            u_p[sw][p] = u_val
            
            if 0 < u_val < 1:
                term = u_val / (2 * (1 - u_val))
                w_hol = (T_PROC_SWITCH * term) / T_MAX
                
                # Asignar a enlaces L3->L4 correspondientes
                prio_node = f"L4_Priority_{sw}_Prio_{p}"
                if prio_node in labels:
                    idx_dst = labels.index(prio_node)
                    # Buscar orígenes L3 conectados
                    for idx_src in np.where(A[:, idx_dst] == 1)[0]:
                        W[idx_src, idx_dst] = w_hol
                        # Jitter estimado
                        J[idx_src, idx_dst] = w_hol * np.sqrt((2 - u_val)/u_val)

    # --- 4. L4/Trunk -> L5 (Colas de Salida M/M/1) ---
    # Acumular tráfico por puerto de salida
    port_stats = {} # {label_puerto: {prio: {bps, ssf, sf}}}
    
    for vlan, df in D.items():
        prio = get_vlan_priority(vlan)
        for src in df.index:
            for dst in df.columns:
                bw = df.at[src, dst]
                if bw > 0:
                    sw_s, sw_d = sw_map[src], sw_map[dst]
                    
                    # Puertos afectados
                    ports = []
                    if sw_s == sw_d:
                        ports.append(f"L5_OutPort_{dst}")
                    else:
                        ports.append(f"L4_Trunk_{sw_s}_to_{sw_d}")
                        ports.append(f"L5_OutPort_{dst}")
                    
                    # Specs simplificados para cálculo
                    specs = TRAFFIC_SPECS["MON"]
                    if "SV" in vlan: specs = TRAFFIC_SPECS["SV"]
                    elif "GOOSE" in vlan: specs = TRAFFIC_SPECS["GOOSE"]
                    elif "PTP" in vlan: specs = TRAFFIC_SPECS["PTP"]
                    
                    for p_label in ports:
                        if p_label not in port_stats:
                            port_stats[p_label] = {p: {'bps':0, 'ssf':0, 'sf':0} for p in range(8)}
                        
                        port_stats[p_label][prio]['bps'] += (bw * 1e6)
                        port_stats[p_label][prio]['ssf'] += (specs['size'] * specs['freq'])
                        port_stats[p_label][prio]['sf'] += specs['freq']

    # Calcular pesos de cola
    for p_label, stats in port_stats.items():
        if p_label not in labels: continue
        idx_dst = labels.index(p_label)
        
        cap = CAPACITY_TRUNK if "Trunk" in p_label else CAPACITY_DEFAULT
        w_accum = 0.0
        
        for k in range(7, -1, -1):
            data = stats[k]
            if data['sf'] > 0:
                l_prom = (data['ssf'] / data['sf']) * 8
                w_trans = (l_prom / cap) / T_MAX
                
                # Carga acumulada (Strict Priority)
                bps_accum = sum(stats[p]['bps'] for p in range(k, 8))
                rho = bps_accum / cap
                
                w_cola = 0.0
                p_loss = 0.0
                
                if 0 < rho < 1.0:
                    delay_s = (rho * l_prom) / (FACTOR_16 * cap * (1 - rho))
                    w_cola = delay_s / T_MAX
                    # Pérdida M/M/1/K
                    p_loss = (rho**K_BUFFER * (1 - rho)) / (1 - rho**(K_BUFFER + 1))
                elif rho >= 1.0:
                    w_cola = 1.0 # Saturación
                    p_loss = 1.0 # Pérdida total
                
                w_total = min(1.0, w_trans + w_cola + w_accum)
                w_accum = w_total
                
                # Asignar a enlaces entrantes (L4 -> Puerto)
                # Necesitamos saber qué switch es para buscar el nodo de prioridad correcto
                # Simplificación: Buscamos en A quién conecta a este puerto con prioridad k
                for idx_src in np.where(A[:, idx_dst] == 1)[0]:
                    src_lbl = labels[idx_src]
                    if f"_Prio_{k}" in src_lbl:
                        W[idx_src, idx_dst] = w_total
                        if rho > 0 and rho < 1:
                            J[idx_src, idx_dst] = w_cola * np.sqrt((2-rho)/rho)
                        PL[idx_src, idx_dst] = p_loss

    # --- 5. L5 -> L1 (Físico Salida) ---
    # Similar a L1->L2 pero de salida (omitiendo detalles por brevedad, asumiendo simetría aprox)
    for d in all_devices:
        if d != 'GPS':
            try:
                idx_src = labels.index(f"L5_OutPort_{d}")
                idx_dst = labels.index(f"L1_Equipment_{d}")
                # Usamos un peso base pequeño si no hay cálculo detallado
                W[idx_src, idx_dst] = (100 * 8 / CAPACITY_DEFAULT) / T_MAX 
            except ValueError: pass

    return W, J, PL, u_p

# =================================================================
# 5. ANÁLISIS Y MÉTRICAS
# =================================================================

def analizar_red(N, labels, A, W, J, PL, u_p, resultados_v):
    """Recopila todas las métricas para el reporte."""
    
    # 1. Métricas de Grafos (NetworkX)
    G = nx.from_numpy_array(A) # Grafo simple para estructura
    try:
        evs = np.real(np.linalg.eigvals(A))
        spectral_radius = max(abs(evs))
        # Conectividad Algebraica (Fiedler) sobre grafo ponderado no dirigido
        G_weighted = nx.from_numpy_array((W + W.T)/2)
        lambda_2 = nx.algebraic_connectivity(G_weighted, weight='weight', method='tracemin')
    except:
        spectral_radius, lambda_2 = 0.0, 0.0

    # --- NUEVAS MÉTRICAS DE GRAFOS COMPLEJOS ---
    try:
        # Verificar conectividad para métricas de camino
        if nx.is_connected(G):
            diameter = nx.diameter(G)
            avg_path = nx.average_shortest_path_length(G)
        else:
            # Si el grafo no es conexo, usar la componente gigante
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            diameter = nx.diameter(subgraph)
            avg_path = nx.average_shortest_path_length(subgraph)
            
        edge_conn = nx.edge_connectivity(G)
        # Centralidad (Max Betweenness) - Identifica cuellos de botella
        bet_cen = nx.betweenness_centrality(G)
        max_bet_cen = max(bet_cen.values()) if bet_cen else 0
    except Exception as e:
        print(f"Advertencia calculando métricas complejas: {e}")
        diameter, avg_path, edge_conn, max_bet_cen = 0, 0, 0, 0

    # 2. Latencias y Pérdidas por Prioridad
    stats_prio = {p: {'lat': [], 'jit': [], 'loss': []} for p in [4, 5, 6, 7]}
    
    for i, label in enumerate(labels):
        if "L4_Priority_" in label and "Trunk" not in label:
            try:
                p = int(label.split('_')[-1])
                # Mirar enlaces salientes
                indices_out = np.where(W[i, :] > 0)[0]
                for j in indices_out:
                    stats_prio[p]['lat'].append(W[i, j] * T_MAX * 1000) # ms
                    stats_prio[p]['jit'].append(J[i, j] * T_MAX * 1000) # ms
                    stats_prio[p]['loss'].append(PL[i, j] * 100)        # %
            except: pass

    # 3. Consolidar
    metrics = {
        'N_Bahias': N,
        'Total_Nodos': len(labels),
        'Fiedler_Lambda2': lambda_2,
        'Spectral_Radius': spectral_radius,
        'Diameter': diameter,
        'Edge_Connectivity': edge_conn,
        'Max_Betweenness': max_bet_cen,
        'Avg_Shortest_Path': avg_path,
        'Max_Util_SW1': max(u_p['SW1'].values()) if 'SW1' in u_p else 0,
        'Max_Util_SW2': max(u_p['SW2'].values()) if 'SW2' in u_p else 0,
    }
    
    for p in [4, 5, 6, 7]:
        lats = stats_prio[p]['lat']
        metrics[f'Latencia_Avg_P{p}'] = np.mean(lats) if lats else 0
        metrics[f'Latencia_Max_P{p}'] = np.max(lats) if lats else 0
        metrics[f'Jitter_Avg_P{p}'] = np.mean(stats_prio[p]['jit']) if lats else 0
        metrics[f'Loss_Avg_P{p}'] = np.mean(stats_prio[p]['loss']) if lats else 0

    return metrics

# =================================================================
# 6. FUNCIÓN PRINCIPAL (MAIN LOOP)
# =================================================================

def run_simulation(max_bays=10):
    historico = []
    
    print(f"Iniciando simulación para 1 a {max_bays} bahías...")
    
    for N in range(1, max_bays + 1):
        print(f"Simulando N={N}...")
        
        # 1. Generar Modelo
        devs, sw_map = generar_dispositivos(N)
        D = generar_tensor_demanda(N, devs)
        
        # 2. Construir Topología
        labels, A = construir_topologia(N, devs, D, sw_map)
        
        # 3. Calcular Física (Pesos)
        W, J, PL, u_p = calcular_pesos(N, labels, A, D, sw_map, devs)
        
        # 4. Analizar
        # (Simplificación: No recalculamos vulnerabilidad V en cada paso para velocidad,
        #  pero se podría añadir llamando a funciones específicas de NetworkX)
        metrics = analizar_red(N, labels, A, W, J, PL, u_p, [])
        
        historico.append(metrics)
        
    return pd.DataFrame(historico)

def exportar_reporte_organizado(df, filename="Reporte_Escalabilidad_Final.xlsx"):
    """
    Exporta los resultados a Excel con múltiples hojas y formato organizado.
    Incluye evaluación de cumplimiento de límites IEC 61850.
    """
    print(f"Generando reporte organizado en {filename}...")
    
    # 1. Crear DataFrame de Resumen (KPIs Principales)
    resumen = pd.DataFrame()
    resumen['N_Bahias'] = df['N_Bahias']
    
    # Latencias Críticas (P6=Protección, P4=SV)
    if 'Latencia_Max_P6' in df.columns:
        resumen['Latencia_Max_Prot_P6 (ms)'] = df['Latencia_Max_P6']
    if 'Latencia_Max_P4' in df.columns:
        resumen['Latencia_Max_SV_P4 (ms)'] = df['Latencia_Max_P4']
        
    # Utilización Máxima de Switches (Convertir a %)
    cols_util = [c for c in df.columns if 'Max_Util' in c]
    if cols_util:
        resumen['Max_Switch_Util (%)'] = df[cols_util].max(axis=1) * 100
        
    # Estado de Cumplimiento (Pass/Fail)
    # Criterio: Latencia < 3ms para tráfico crítico
    def check_compliance(row):
        lat_p6 = row.get('Latencia_Max_P6', 0)
        lat_p4 = row.get('Latencia_Max_P4', 0)
        loss_p6 = row.get('Loss_Avg_P6', 0)
        
        if lat_p6 > 3.0 or lat_p4 > 3.0:
            return "FAIL (Latencia > 3ms)"
        if loss_p6 > 0.01:
            return "FAIL (Pérdida Paquetes)"
        return "PASS"
        
    resumen['Estado_IEC61850'] = df.apply(check_compliance, axis=1)

    # 2. Separar métricas por categoría
    cols_lat = [c for c in df.columns if 'Latencia' in c]
    cols_jit = [c for c in df.columns if 'Jitter' in c]
    cols_loss = [c for c in df.columns if 'Loss' in c]
    cols_topo = ['Total_Nodos', 'Fiedler_Lambda2', 'Spectral_Radius', 
                 'Diameter', 'Edge_Connectivity', 'Max_Betweenness', 'Avg_Shortest_Path'] + cols_util
    # Filtrar columnas que realmente existen
    cols_topo = [c for c in cols_topo if c in df.columns]

    # 3. Escribir a Excel
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            resumen.to_excel(writer, sheet_name='Resumen Ejecutivo', index=False)
            if cols_lat: df[['N_Bahias'] + cols_lat].to_excel(writer, sheet_name='Latencia (ms)', index=False)
            if cols_jit: df[['N_Bahias'] + cols_jit].to_excel(writer, sheet_name='Jitter (ms)', index=False)
            if cols_loss: df[['N_Bahias'] + cols_loss].to_excel(writer, sheet_name='Pérdida (%)', index=False)
            if cols_topo: df[['N_Bahias'] + cols_topo].to_excel(writer, sheet_name='Topología y Carga', index=False)
            df.to_excel(writer, sheet_name='Raw Data', index=False)
    except Exception as e:
        print(f"Advertencia: No se pudo usar formato avanzado ({e}). Usando exportación simple.")
        df.to_excel(filename, index=False)

def plot_results(df):
    """Generates final plots in English with professional styling."""
    if df.empty: return

    # Configuración de estilo
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a 5x2 grid for comprehensive analysis (Expanded for new metrics)
    fig, axes = plt.subplots(5, 2, figsize=(16, 25))
    fig.suptitle(f"IEC 61850 Substation Scalability Analysis (N=1 to {df['N_Bahias'].max()})", fontsize=16, weight='bold')

    # Colors for consistency
    color_p4 = '#1f77b4' # Blue for SV (Sampled Values)
    color_p6 = '#d62728' # Red for Protection (GOOSE)
    
    # --- 1. Latency (Critical) ---
    ax = axes[0, 0]
    # P6 (Protection)
    ax.plot(df['N_Bahias'], df['Latencia_Max_P6'], color=color_p6, marker='o', linestyle='-', label='Prot (P6) Max')
    ax.plot(df['N_Bahias'], df['Latencia_Avg_P6'], color=color_p6, marker='x', linestyle='--', alpha=0.7, label='Prot (P6) Avg')
    # P4 (SV)
    ax.plot(df['N_Bahias'], df['Latencia_Max_P4'], color=color_p4, marker='s', linestyle='-', label='SV (P4) Max')
    
    # IEC 61850 Limit
    ax.axhline(3.0, color='black', linestyle=':', linewidth=2, label='IEC 61850 Limit (3ms)')
    
    ax.set_title("End-to-End Latency", fontsize=12, weight='bold')
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Number of Bays")
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)

    # --- 2. Jitter (Variation) ---
    ax = axes[0, 1]
    ax.plot(df['N_Bahias'], df['Jitter_Avg_P6'], color=color_p6, marker='o', label='Prot (P6) Jitter')
    ax.plot(df['N_Bahias'], df['Jitter_Avg_P4'], color=color_p4, marker='s', label='SV (P4) Jitter')
    
    ax.set_title("Packet Jitter (Average)", fontsize=12, weight='bold')
    ax.set_ylabel("Jitter (ms)")
    ax.set_xlabel("Number of Bays")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.7)

    # --- 3. Packet Loss ---
    ax = axes[1, 0]
    ax.plot(df['N_Bahias'], df['Loss_Avg_P6'], color=color_p6, marker='o', label='Prot (P6) Loss')
    ax.plot(df['N_Bahias'], df['Loss_Avg_P4'], color=color_p4, marker='s', label='SV (P4) Loss')
    
    ax.set_title("Packet Loss Probability", fontsize=12, weight='bold')
    ax.set_ylabel("Loss (%)")
    ax.set_xlabel("Number of Bays")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.7)
    
    # --- 4. Switch Utilization ---
    ax = axes[1, 1]
    ax.plot(df['N_Bahias'], df['Max_Util_SW1']*100, color='purple', marker='D', label='Core Switch 1')
    if 'Max_Util_SW2' in df.columns:
        ax.plot(df['N_Bahias'], df['Max_Util_SW2']*100, color='orange', marker='D', label='Core Switch 2')
    
    ax.axhline(100, color='red', linestyle='-', linewidth=1, label='Saturation (100%)')
    
    ax.set_title("Switch Backplane Utilization", fontsize=12, weight='bold')
    ax.set_ylabel("Utilization (%)")
    ax.set_xlabel("Number of Bays")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.7)

    # --- 5. Network Robustness (Topology) ---
    ax = axes[2, 0]
    ax.plot(df['N_Bahias'], df['Fiedler_Lambda2'], color='green', marker='^', linestyle='-')
    
    ax.set_title("Algebraic Connectivity (Robustness)", fontsize=12, weight='bold')
    ax.set_ylabel("Fiedler Value (Lambda 2)")
    ax.set_xlabel("Number of Bays")
    ax.grid(True, alpha=0.7)
    
    # --- 6. Edge Connectivity (Redundancy) ---
    ax = axes[2, 1]
    ax.plot(df['N_Bahias'], df['Edge_Connectivity'], color='teal', marker='s', linestyle='-')
    
    ax.set_title("Edge Connectivity (Redundancy)", fontsize=12, weight='bold')
    ax.set_ylabel("Min Cuts to Failure")
    ax.set_xlabel("Number of Bays")
    ax.grid(True, alpha=0.7)

    # --- 7. Network Diameter ---
    ax = axes[3, 0]
    ax.plot(df['N_Bahias'], df['Diameter'], color='brown', marker='o', linestyle='-')
    
    ax.set_title("Network Diameter (Max Hops)", fontsize=12, weight='bold')
    ax.set_ylabel("Hops")
    ax.set_xlabel("Number of Bays")
    ax.grid(True, alpha=0.7)

    # --- 8. Average Shortest Path ---
    ax = axes[3, 1]
    ax.plot(df['N_Bahias'], df['Avg_Shortest_Path'], color='magenta', marker='x', linestyle='--')
    
    ax.set_title("Average Shortest Path Length", fontsize=12, weight='bold')
    ax.set_ylabel("Avg Hops")
    ax.set_xlabel("Number of Bays")
    ax.grid(True, alpha=0.7)

    # --- 9. Max Betweenness Centrality (Bottleneck) ---
    ax = axes[4, 0]
    ax.plot(df['N_Bahias'], df['Max_Betweenness'], color='red', marker='D', linestyle='-')
    
    ax.set_title("Max Betweenness Centrality (Bottleneck Stress)", fontsize=12, weight='bold')
    ax.set_ylabel("Centrality Score")
    ax.set_xlabel("Number of Bays")
    ax.grid(True, alpha=0.7)

    # --- 10. Network Complexity (Nodes) ---
    ax = axes[4, 1]
    ax.plot(df['N_Bahias'], df['Total_Nodos'], color='gray', marker='.', linestyle='-')
    ax.set_title("Network Complexity (Graph Size)", fontsize=12, weight='bold')
    ax.set_ylabel("Total Nodes (Graph)")
    ax.set_xlabel("Number of Bays")
    ax.grid(True, alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    
    output_file = "Results_Scalability_English.png"
    plt.savefig(output_file, dpi=300)
    plt.show()
    print(f"Graphics generated: {output_file}")

# =================================================================
# EJECUCIÓN
# =================================================================

if __name__ == "__main__":
    df_resultados = run_simulation(max_bays=10)
    print("\nResultados Finales:")
    print(df_resultados.head())
    
    # Exportar
    exportar_reporte_organizado(df_resultados, "Reporte_Escalabilidad_Final.xlsx")
    
    # Graficar
    plot_results(df_resultados)
