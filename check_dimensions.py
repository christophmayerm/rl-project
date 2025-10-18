#!/usr/bin/env python3
"""
Script to check the actual state space dimensions and memory requirements
for the racetrack environment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.racetrack_simulator import RaceTrackConfigurableEnv

def check_dimensions(track_file='T1'):
    """Check dimensions for a given track file."""

    print(f"=" * 70)
    print(f"Checking dimensions for track: {track_file}")
    print(f"=" * 70)

    # Initialize environment
    mdp = RaceTrackConfigurableEnv(
        track_file=track_file,
        initial_configuration=[0.5, 0.5, 0, 0],
        pfail=0.07
    )

    # State space info
    print(f"\n📊 STATE SPACE:")
    print(f"  - Track grid dimensions: {mdp.nrow} × {mdp.ncol}")
    print(f"  - Valid track positions (nlin): {mdp.nlin}")
    print(f"  - Velocity values: {mdp.vel}")
    print(f"  - Number of velocity options (nvel): {mdp.nvel}")
    print(f"  - Total states (nS): {mdp.nS}")
    print(f"    Formula: nlin × nvel × nvel + 1 = {mdp.nlin} × {mdp.nvel} × {mdp.nvel} + 1 = {mdp.nS}")

    # Action space info
    print(f"\n🎮 ACTION SPACE:")
    print(f"  - Number of actions (nA): {mdp.nA}")
    print(f"    Actions: 0=KEEP, 1=INCx, 2=INCy, 3=DECx, 4=DECy")

    # Combined dimensions
    nSA = mdp.nS * mdp.nA
    print(f"\n🔢 COMBINED DIMENSIONS:")
    print(f"  - State-Action pairs (nSA = nS × nA): {nSA}")

    # Matrix dimensions
    print(f"\n📐 MATRIX DIMENSIONS:")
    print(f"  - Q-function: ({nSA}, {nSA})")
    print(f"  - V-function: ({mdp.nS}, {mdp.nS})")
    print(f"  - Model P: ({nSA}, {mdp.nS})")
    print(f"  - Policy π: ({mdp.nS}, {nSA})")

    # Memory requirements (assuming float64 = 8 bytes)
    bytes_per_element = 8
    print(f"\n💾 MEMORY REQUIREMENTS (float64):")

    q_memory_mb = (nSA ** 2) * bytes_per_element / (1024 ** 2)
    v_memory_mb = (mdp.nS ** 2) * bytes_per_element / (1024 ** 2)
    p_memory_mb = (nSA * mdp.nS) * bytes_per_element / (1024 ** 2)
    pi_memory_mb = (mdp.nS * nSA) * bytes_per_element / (1024 ** 2)

    print(f"  - Q-function: {nSA:,} × {nSA:,} = {nSA**2:,} elements = {q_memory_mb:.2f} MB")
    print(f"  - V-function: {mdp.nS:,} × {mdp.nS:,} = {mdp.nS**2:,} elements = {v_memory_mb:.2f} MB")
    print(f"  - Model P: {nSA:,} × {mdp.nS:,} = {nSA * mdp.nS:,} elements = {p_memory_mb:.2f} MB")
    print(f"  - Policy π: {mdp.nS:,} × {nSA:,} = {mdp.nS * nSA:,} elements = {pi_memory_mb:.2f} MB")
    print(f"  - TOTAL (worst case): {q_memory_mb + v_memory_mb + p_memory_mb + pi_memory_mb:.2f} MB")

    # Computational complexity
    print(f"\n⚙️  COMPUTATIONAL COMPLEXITY (per iteration):")
    q_solve_ops = nSA ** 3
    v_solve_ops = mdp.nS ** 3
    p_pi_mult_ops = nSA * nSA * mdp.nS

    print(f"  - Q-function solve: O(nSA³) = {nSA}³ ≈ {q_solve_ops:,.0f} ops ({q_solve_ops/1e9:.2f} billion)")
    print(f"  - V-function solve: O(nS³) = {mdp.nS}³ ≈ {v_solve_ops:,.0f} ops ({v_solve_ops/1e6:.2f} million)")
    print(f"  - P·π multiply: O(nSA² × nS) ≈ {p_pi_mult_ops:,.0f} ops ({p_pi_mult_ops/1e9:.2f} billion)")
    total_ops = q_solve_ops + 2 * v_solve_ops + 3 * p_pi_mult_ops
    print(f"  - TOTAL (approx): {total_ops/1e9:.2f} billion ops/iteration")

    # Other parameters
    print(f"\n🎯 OTHER PARAMETERS:")
    print(f"  - Discount factor (gamma): {mdp.gamma}")
    print(f"  - Horizon: {mdp.horizon}")
    print(f"  - Failure probability (pfail): {mdp.pfail}")

    print(f"\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    # Check dimensions for the track used in simulations
    track_files = ['T1']  # Add more track files if needed

    for track in track_files:
        try:
            check_dimensions(track)
        except Exception as e:
            print(f"Error loading track {track}: {e}\n")
