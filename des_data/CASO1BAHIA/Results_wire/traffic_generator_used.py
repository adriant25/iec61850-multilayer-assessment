# traffic_generator.py

import numpy as np

def generate_sv_packets(sv_flows_df, packets_dict, simulation_time, frequency_hz=4800):
    """
    Generates Sampled Value (SV) packets for each flow defined in the DataFrame.

    Args:
        sv_flows_df (pd.DataFrame): DataFrame containing SV flow definitions.
        packets_dict (dict): Dictionary to store the generated packets, keyed by source equipment.
        simulation_time (float): Total simulation time in seconds.
        frequency_hz (int): The sampling frequency for SV packets.
    """
    time_step = 1 / frequency_hz

    for _, row in sv_flows_df.iterrows():
        flow_id = row["FLOW ID"]
        source = row["SOURCE"]
        size = row["BYTES"]
        vlan = row["VLAN"]
        packet_priority = row["PRIORITY"]

        # Generate timestamps for the packets with a random initial offset.
        initial_time = np.random.uniform(0, time_step)
        timestamps = np.arange(initial_time, simulation_time, time_step)

        # Use numpy to efficiently create arrays for all packet fields.
        num_packets = len(timestamps)
        flow_ids = np.full(num_packets, flow_id)
        packet_types = np.full(num_packets, "SV", dtype=object)
        source_list = np.full(num_packets, source, dtype=object)
        sizes = np.full(num_packets, size)
        vlans = np.full(num_packets, vlan)
        priorities = np.full(num_packets, packet_priority)
        sequences = np.tile(np.arange(frequency_hz), num_packets // frequency_hz + 1)[:num_packets]
        routes = [['Cre_' + source] for _ in range(num_packets)]

        # Assemble packet data as a list of columns.
        packet_data = [
            sizes.tolist(),
            vlans.tolist(),
            priorities.tolist(),
            source_list.tolist(),
            packet_types.tolist(),
            flow_ids.tolist(),
            routes,
            sequences.tolist(),
            timestamps.tolist()
        ]

        # Add the generated packets to the source equipment's list and sort by time.
        packets_dict[source].extend(zip(*packet_data))
        packets_dict[source].sort(key=lambda p: p[-1])


def generate_goose_packets(goose_flows_df, packets_dict, dest_flow_info, source_flow_info, simulation_time):
    """
    Generates initial GOOSE heartbeat packets and stores flow information.

    Args:
        goose_flows_df (pd.DataFrame): DataFrame with GOOSE flow definitions.
        packets_dict (dict): Dictionary to store the generated packets.
        dest_flow_info (dict): Dictionary to store flow info per destination.
        source_flow_info (dict): Dictionary to store general flow info at the source.
        simulation_time (float): Total simulation time.
    """
    for _, row in goose_flows_df.iterrows():
        flow_id = row["FLOW ID"]
        source = row["SOURCE"]
        destinations = row["DESTINATION(S)"]
        size = row["BYTES"]
        vlan = row["VLAN"]
        packet_priority = row["PRIORITY"]
        t0 = row["T0 (Tmax s)"]
        t1 = row["T1 (Tmin ms)"]

        # Store general flow information at the source equipment.
        source_flow_info.setdefault(source, {})[flow_id] = [1, t1, t0]

        # Store flow information for each destination equipment.
        for dest in destinations:
            dest_flow_info.setdefault(dest, {})[flow_id] = [1, t1, t0]

        # Generate timestamps for the heartbeat packets.
        initial_time = np.random.uniform(0, t0)
        timestamps = np.arange(initial_time, simulation_time, t0)

        # Generate fields for each packet.
        num_packets = len(timestamps)
        flow_ids = np.full(num_packets, flow_id)
        packet_types = np.full(num_packets, "GOOSE", dtype=object)
        source_list = np.full(num_packets, source, dtype=object)
        sizes = np.full(num_packets, size)
        vlans = np.full(num_packets, vlan)
        priorities = np.full(num_packets, packet_priority)
        states = np.full(num_packets, 1)
        sequences = np.tile(np.arange(256), num_packets // 256 + 1)[:num_packets]
        routes = [['Cre_' + source] for _ in range(num_packets)]

        packet_data = [
            sizes.tolist(),
            vlans.tolist(),
            priorities.tolist(),
            source_list.tolist(),
            packet_types.tolist(),
            flow_ids.tolist(),
            routes,
            states.tolist(),
            sequences.tolist(),
            timestamps.tolist()
        ]

        # Add packets to the source equipment's list and sort by time.
        packets_dict[source].extend(zip(*packet_data))
        packets_dict[source].sort(key=lambda p: p[-1])


def generate_ptp_packets(simulation_time, vlan, priority, packets_dict, bus_suffix=""):
    """
    Generates PTP (Precision Time Protocol) packets for any equipment with 'GPS' in its name.

    Args:
        simulation_time (float): Total simulation time.
        vlan (int): VLAN ID for the PTP packets.
        priority (int): Priority for the PTP packets.
        packets_dict (dict): Dictionary to store the generated packets.
        bus_suffix (str): Suffix to identify the bus (e.g., "_SB") for logging.
    """
    # Generate base timestamps with an offset for each message type.
    t0 = np.random.uniform(0, 1)
    timestamps = {
        "Announce": np.arange(t0, simulation_time, 1),
        "Sync": np.arange(t0 + 0.66, simulation_time, 1),
        "Follow_Up": np.arange(t0 + 0.661, simulation_time, 1)
    }

    # Define properties for each PTP message type.
    message_types = {
        "Announce": "PTP Announce Message",
        "Sync": "PTP Sync Message",
        "Follow_Up": "PTP Follow_Up Message"
    }
    message_sizes = {"Announce": 120, "Sync": 80, "Follow_Up": 78}  # + 20 B wire overhead

    found_gps_count = 0

    # Find all equipment acting as a GPS source and generate packets.
    for equipment_name in packets_dict:
        if "GPS" in equipment_name:
            gps_packets = []
            for msg_type, ts in timestamps.items():
                num_packets = len(ts)
                packet_data = [
                    np.full(num_packets, message_sizes[msg_type]).tolist(),
                    np.full(num_packets, vlan).tolist(),
                    np.full(num_packets, priority).tolist(),
                    np.full(num_packets, equipment_name, dtype=object).tolist(),
                    np.full(num_packets, message_types[msg_type], dtype=object).tolist(),
                    np.arange(num_packets).tolist(),
                    [["Cre_" + equipment_name] for _ in ts],
                    ts.tolist()
                ]
                gps_packets.extend(zip(*packet_data))

            # Sort all generated PTP packets for this GPS by time and assign them.
            gps_packets.sort(key=lambda p: p[-1])
            packets_dict[equipment_name] = gps_packets
            found_gps_count += 1

    bus_name = 'SB' if bus_suffix else 'PB'
    if found_gps_count == 0:
        print(f"\nWarning: No equipment with 'GPS' in its name was found on the {bus_name} bus.")
    else:
        print(f"\nSuccessfully generated PTP packets for {found_gps_count} GPS source(s) on the {bus_name} bus.")