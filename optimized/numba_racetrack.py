"""
Numba-optimized functions for the RaceTrack environment.
These functions provide significant speedup for the most computationally intensive operations.
"""

import numpy as np
from numba import njit, prange, types
from numba.typed import Dict, List


@njit(cache=True)
def check_valid_state_numba(x, y, nrow, ncol, track_flat, track_ncol):
    """
    Numba-optimized state validity checking.

    Args:
        x, y: Position coordinates
        nrow, ncol: Track dimensions
        track_flat: Flattened track array
        track_ncol: Number of columns in track

    Returns:
        bool: True if state is valid, False otherwise
    """
    # Check bounds
    if x < 0 or x >= nrow or y < 0 or y >= ncol:
        return False

    # Get track value at position
    track_val = track_flat[x * track_ncol + y]

    # Check if blank (space) or wall ('4')
    if track_val == ord(' ') or track_val == ord('4'):
        return False

    return True


@njit(cache=True)
def check_valid_path_numba(x1, y1, x2, y2, nrow, ncol, track_flat, track_ncol):
    """
    Numba-optimized path validity checking.

    Args:
        x1, y1: Start position
        x2, y2: End position
        nrow, ncol: Track dimensions
        track_flat: Flattened track array
        track_ncol: Number of columns in track

    Returns:
        bool: True if path is valid (no collision), False if collision
    """
    step = 0.1

    # Check points along the path
    k = step
    while k < 1.0:
        # Linear interpolation
        px = k * x2 + (1 - k) * x1
        py = k * y2 + (1 - k) * y1

        # Floor to get integer coordinates
        px_int = int(np.floor(px))
        py_int = int(np.floor(py))

        # Check if this point is valid (note: inverted logic from original)
        if not check_valid_state_numba(px_int, py_int, nrow, ncol, track_flat, track_ncol):
            return False

        k += step

    return True


@njit(cache=True)
def next_state_numba(x, y, vx, vy, a, outcome, nrow, ncol, track_flat, track_ncol):
    """
    Numba-optimized next state computation.

    Args:
        x, y: Current position
        vx, vy: Current velocity
        a: Action (0=KEEP, 1=INCx, 2=INCy, 3=DECx, 4=DECy)
        outcome: Success (1) or failure (0)
        nrow, ncol: Track dimensions
        track_flat: Flattened track array
        track_ncol: Number of columns in track

    Returns:
        tuple: (nx, ny, nvx, nvy) - next state
    """
    if a == 0 or outcome == 0:  # keep or failed action
        nvx = vx
        nvy = vy
    elif a == 1:  # increment x
        nvx = vx + 1
        nvy = vy
    elif a == 2:  # increment y
        nvx = vx
        nvy = vy + 1
    elif a == 3:  # decrement x
        nvx = vx - 1
        nvy = vy
    elif a == 4:  # decrement y
        nvx = vx
        nvy = vy - 1
    else:
        nvx = vx
        nvy = vy

    nx = x + nvx
    ny = y + nvy

    # Check validity of the next state
    if not check_valid_state_numba(nx, ny, nrow, ncol, track_flat, track_ncol):
        return (x, y, 0, 0)

    # Check path validity with different offsets
    if not check_valid_path_numba(x + 0.5, y + 0.5, nx + 0.5, ny, nrow, ncol, track_flat, track_ncol):
        return (x, y, 0, 0)
    if not check_valid_path_numba(x + 0.5, y + 0.5, nx, ny + 0.5, nrow, ncol, track_flat, track_ncol):
        return (x, y, 0, 0)
    if not check_valid_path_numba(x + 0.5, y + 0.5, nx, ny, nrow, ncol, track_flat, track_ncol):
        return (x, y, 0, 0)
    if not check_valid_path_numba(x + 0.5, y + 0.5, nx + 0.5, ny + 0.5, nrow, ncol, track_flat, track_ncol):
        return (x, y, 0, 0)

    return (nx, ny, nvx, nvy)


@njit(cache=True)
def s_to_i_numba(x, y, vx, vy, lin, nvel, min_vel):
    """
    Numba-optimized state to index conversion.

    Args:
        x, y: Position
        vx, vy: Velocity
        lin: Linear indices array
        nvel: Number of velocity states
        min_vel: Minimum velocity

    Returns:
        int: State index
    """
    # Find the linear index for position (x, y)
    s_lin = -1
    for i in range(len(lin)):
        if lin[i, 0] == x and lin[i, 1] == y:
            s_lin = i
            break

    if s_lin == -1:
        return -1  # Invalid state

    index = s_lin * nvel * nvel + (vx - min_vel) * nvel + (vy - min_vel)
    return index


@njit(cache=True)
def reward_state_numba(x, y, vx, vy, weight, track_flat, track_ncol, reward_fail_abs):
    """
    Numba-optimized reward computation.

    Args:
        x, y: Position
        vx, vy: Velocity
        weight: Reward weight vector
        track_flat: Flattened track array
        track_ncol: Number of columns in track
        reward_fail_abs: Absolute failure reward

    Returns:
        float: Reward value
    """
    if x == -1:
        return reward_fail_abs

    # Get track type
    track_val = track_flat[x * track_ncol + y]

    speed = vx * vx + vy * vy

    # Compute basis features
    isGoal = 1.0 if track_val == ord('2') else 0.0
    isOffroad = 1.0 if track_val == ord('3') else 0.0
    isOnTrack = 1.0 if track_val == ord('5') else 0.0
    isZeroSpeed = 1.0 if speed == 0 else 0.0
    isLowSpeed = 1.0 if speed < 2 else 0.0
    isHighSpeed = 1.0 if speed >= 2 else 0.0

    # Compute weighted sum
    reward = (weight[0] * isGoal +
              weight[1] * isOffroad +
              weight[2] * (isZeroSpeed * isOnTrack) +
              weight[3] * (isLowSpeed * isOnTrack) +
              weight[4] * (isHighSpeed * isOnTrack))

    return reward


@njit(cache=True)
def build_P_matrices_numba(
    lin, vel, nA, nS,
    track_flat, track_ncol, nrow, ncol,
    reward_weight, reward_fail_abs,
    max_psuc, min_psuc, max_psuc2, min_psuc2,
    max_pboost, min_pboost, pfail, max_speed,
    min_vel, max_vel, min_vel_nb, max_vel_nb, nvel
):
    """
    Numba-optimized version of _build_P() function.
    Builds all 4 vertex transition models in parallel.

    Returns:
        Four dictionaries representing the vertex models
    """

    # Pre-allocate probability matrices (using dense representation for speed)
    # Each matrix is (nS, nA, nS) but we'll use sparse representation later
    P_hs_nb = np.zeros((nS, nA, nS), dtype=np.float64)
    P_ls_nb = np.zeros((nS, nA, nS), dtype=np.float64)
    P_hs_b = np.zeros((nS, nA, nS), dtype=np.float64)
    P_ls_b = np.zeros((nS, nA, nS), dtype=np.float64)

    # Process each position
    for pos_idx in range(len(lin)):
        x, y = lin[pos_idx, 0], lin[pos_idx, 1]

        for vx_idx in range(len(vel)):
            vx = vel[vx_idx]
            for vy_idx in range(len(vel)):
                vy = vel[vy_idx]

                s = s_to_i_numba(x, y, vx, vy, lin, nvel, min_vel)
                if s == -1:
                    continue

                speed = vx * vx + vy * vy

                # Get track type
                track_type = track_flat[x * track_ncol + y]

                if track_type == ord('2'):  # Goal state
                    for a in range(nA):
                        P_hs_nb[s, a, s] = 1.0
                        P_ls_nb[s, a, s] = 1.0
                        P_hs_b[s, a, s] = 1.0
                        P_ls_b[s, a, s] = 1.0
                else:
                    # Valid actions
                    valid_actions = np.zeros(nA, dtype=np.int32)
                    valid_actions[0] = 1  # KEEP always valid
                    if vx < max_vel:
                        valid_actions[1] = 1  # INCx
                    if vx > min_vel:
                        valid_actions[3] = 1  # DECx
                    if vy < max_vel:
                        valid_actions[2] = 1  # INCy
                    if vy > min_vel:
                        valid_actions[4] = 1  # DECy

                    # Valid actions without boost
                    valid_actions_nb = np.zeros(nA, dtype=np.int32)
                    valid_actions_nb[0] = 1  # KEEP always valid
                    if vx < max_vel_nb:
                        valid_actions_nb[1] = 1  # INCx
                    if vx > min_vel_nb:
                        valid_actions_nb[3] = 1  # DECx
                    if vy < max_vel_nb:
                        valid_actions_nb[2] = 1  # INCy
                    if vy > min_vel_nb:
                        valid_actions_nb[4] = 1  # DECy

                    for a in range(nA):
                        if valid_actions_nb[a] == 1:
                            # Action can be performed without boost

                            # Success probabilities
                            psuc_hs = min_psuc + ((max_psuc - min_psuc) / max_speed) * speed
                            psuc_ls = max_psuc2 - ((max_psuc2 - min_psuc2) / max_speed) * speed

                            # Compute next state for successful action
                            nx, ny, nvx, nvy = next_state_numba(x, y, vx, vy, a, 1,
                                                                nrow, ncol, track_flat, track_ncol)
                            ns = s_to_i_numba(nx, ny, nvx, nvy, lin, nvel, min_vel)

                            if ns != -1:
                                # No boost models
                                P_hs_nb[s, a, ns] += psuc_hs
                                P_ls_nb[s, a, ns] += psuc_ls

                                # Boost models (with failure possibility)
                                P_hs_b[s, a, ns] += psuc_hs * (1 - pfail)
                                P_ls_b[s, a, ns] += psuc_ls * (1 - pfail)

                                # Failure state
                                P_hs_b[s, a, nS - 1] += psuc_hs * pfail
                                P_ls_b[s, a, nS - 1] += psuc_ls * pfail

                            # Failed action transitions
                            pins_hs = 1 - psuc_hs
                            pins_ls = 1 - psuc_ls

                            # Count other valid actions for failure distribution
                            n_fail_actions = 0
                            for a_fail in range(nA):
                                if valid_actions_nb[a_fail] == 1 and a_fail != a and a_fail != 0:
                                    n_fail_actions += 1

                            if n_fail_actions > 0:
                                for a_fail in range(nA):
                                    if valid_actions_nb[a_fail] == 1 and a_fail != a and a_fail != 0:
                                        nx, ny, nvx, nvy = next_state_numba(x, y, vx, vy, a_fail, 1,
                                                                            nrow, ncol, track_flat, track_ncol)
                                        ns = s_to_i_numba(nx, ny, nvx, nvy, lin, nvel, min_vel)

                                        if ns != -1:
                                            prob_hs = pins_hs / n_fail_actions
                                            prob_ls = pins_ls / n_fail_actions

                                            P_hs_nb[s, a, ns] += prob_hs
                                            P_ls_nb[s, a, ns] += prob_ls
                                            P_hs_b[s, a, ns] += prob_hs
                                            P_ls_b[s, a, ns] += prob_ls

                        elif valid_actions[a] == 1:
                            # Action requires boost capability

                            # Success probabilities
                            psuc_hs = min_psuc + ((max_psuc - min_psuc) / max_speed) * speed
                            psuc_ls = max_psuc2 - ((max_psuc2 - min_psuc2) / max_speed) * speed

                            # No boost state (action 0 = KEEP)
                            nx_nb, ny_nb, nvx_nb, nvy_nb = next_state_numba(x, y, vx, vy, 0, 1,
                                                                            nrow, ncol, track_flat, track_ncol)
                            ns_nb = s_to_i_numba(nx_nb, ny_nb, nvx_nb, nvy_nb, lin, nvel, min_vel)

                            # Boost state
                            nx_b, ny_b, nvx_b, nvy_b = next_state_numba(x, y, vx, vy, a, 1,
                                                                        nrow, ncol, track_flat, track_ncol)
                            ns_b = s_to_i_numba(nx_b, ny_b, nvx_b, nvy_b, lin, nvel, min_vel)

                            # Failure state
                            ns_f = nS - 1

                            pins_hs_b = 1.0
                            pins_ls_b = 1.0
                            pins_hs_nb = 1.0
                            pins_ls_nb = 1.0

                            # High speed boost
                            prob_b = psuc_hs * max_pboost * (1.0 - pfail)
                            prob_f = psuc_hs * max_pboost * pfail
                            prob_nb = psuc_hs * (1 - max_pboost)
                            pins_hs_b = pins_hs_b - prob_b - prob_f - prob_nb

                            if ns_b != -1:
                                P_hs_b[s, a, ns_b] += prob_b
                            P_hs_b[s, a, ns_f] += prob_f
                            if ns_nb != -1:
                                P_hs_b[s, a, ns_nb] += prob_nb

                            # Low speed boost
                            prob_b = psuc_ls * max_pboost * (1.0 - pfail)
                            prob_f = psuc_ls * max_pboost * pfail
                            prob_nb = psuc_ls * (1 - max_pboost)
                            pins_ls_b = pins_ls_b - prob_b - prob_f - prob_nb

                            if ns_b != -1:
                                P_ls_b[s, a, ns_b] += prob_b
                            P_ls_b[s, a, ns_f] += prob_f
                            if ns_nb != -1:
                                P_ls_b[s, a, ns_nb] += prob_nb

                            # High speed no boost
                            prob_b = psuc_hs * min_pboost * (1.0 - pfail)
                            prob_f = psuc_hs * min_pboost * pfail
                            prob_nb = psuc_hs * (1 - min_pboost)
                            pins_hs_nb = pins_hs_nb - prob_b - prob_f - prob_nb

                            if ns_b != -1:
                                P_hs_nb[s, a, ns_b] += prob_b
                            P_hs_nb[s, a, ns_f] += prob_f
                            if ns_nb != -1:
                                P_hs_nb[s, a, ns_nb] += prob_nb

                            # Low speed no boost
                            prob_b = psuc_ls * min_pboost * (1.0 - pfail)
                            prob_f = psuc_ls * min_pboost * pfail
                            prob_nb = psuc_ls * (1 - min_pboost)
                            pins_ls_nb = pins_ls_nb - prob_b - prob_f - prob_nb

                            if ns_b != -1:
                                P_ls_nb[s, a, ns_b] += prob_b
                            P_ls_nb[s, a, ns_f] += prob_f
                            if ns_nb != -1:
                                P_ls_nb[s, a, ns_nb] += prob_nb

                            # Failed action transitions
                            n_fail_actions = 0
                            for a_fail in range(nA):
                                if valid_actions_nb[a_fail] == 1 and a_fail != a and a_fail != 0:
                                    n_fail_actions += 1

                            if n_fail_actions > 0:
                                for a_fail in range(nA):
                                    if valid_actions_nb[a_fail] == 1 and a_fail != a and a_fail != 0:
                                        nx, ny, nvx, nvy = next_state_numba(x, y, vx, vy, a_fail, 1,
                                                                            nrow, ncol, track_flat, track_ncol)
                                        ns = s_to_i_numba(nx, ny, nvx, nvy, lin, nvel, min_vel)

                                        if ns != -1:
                                            P_hs_b[s, a, ns] += pins_hs_b / n_fail_actions
                                            P_ls_b[s, a, ns] += pins_ls_b / n_fail_actions
                                            P_hs_nb[s, a, ns] += pins_hs_nb / n_fail_actions
                                            P_ls_nb[s, a, ns] += pins_ls_nb / n_fail_actions

    # Handle failure state transitions
    for a in range(nA):
        P_hs_nb[nS - 1, a, nS - 1] = 1.0
        P_ls_nb[nS - 1, a, nS - 1] = 1.0
        P_hs_b[nS - 1, a, nS - 1] = 1.0
        P_ls_b[nS - 1, a, nS - 1] = 1.0

    return P_hs_nb, P_ls_nb, P_hs_b, P_ls_b


@njit(cache=True)
def p_sas_to_dict_numba(P_sas, nS, nA, reward_weight, lin, nvel, min_vel,
                         track_flat, track_ncol, reward_fail_abs):
    """
    Convert dense P_sas matrix to sparse dictionary format.
    """
    # We'll return a list of lists for each (s, a) pair
    # Format: For each (s, a), list of (prob, next_state, reward, done)
    result = []

    for s in range(nS):
        for a in range(nA):
            transitions = []
            for s1 in range(nS):
                prob = P_sas[s, a, s1]
                if prob > 0:
                    # Compute reward for next state
                    if s1 == nS - 1:
                        reward = reward_fail_abs
                        done = True
                    else:
                        # Decode state
                        if s1 == nS - 1:
                            x, y, vx, vy = -1, -1, -1, -1
                        else:
                            vy_off = s1 % nvel
                            vy = vy_off + min_vel
                            vx_off = (s1 - vy_off) % (nvel * nvel)
                            vx = (vx_off // nvel) + min_vel
                            s_lin = (s1 - vx_off - vy_off) // (nvel * nvel)
                            x, y = lin[s_lin, 0], lin[s_lin, 1]

                        reward = reward_state_numba(x, y, vx, vy, reward_weight,
                                                   track_flat, track_ncol, reward_fail_abs)
                        track_type = track_flat[x * track_ncol + y] if x != -1 else ord(' ')
                        done = track_type == ord('2')

                    transitions.append((prob, s1, reward, done))

            result.append(transitions)

    return result


def p_sas_numba_wrapper(P_dict, nS, nA):
    """
    Wrapper to convert dictionary format to (nS, nA, nS) matrix.
    This wrapper handles the Python dictionary, then calls the Numba function.

    Args:
        P_dict: Dictionary representation of transitions
        nS: Number of states
        nA: Number of actions

    Returns:
        P_sas: (nS, nA, nS) probability matrix
    """
    # Convert to numpy array format directly
    P_sas = np.zeros((nS, nA, nS), dtype=np.float64)

    for s in range(nS):
        for a in range(nA):
            for prob, s1, _, _ in P_dict[s][a]:
                P_sas[s, a, s1] += prob

    return P_sas


@njit(cache=True)
def p_sa_numba(P_sas, nS, nA):
    """
    Numba-optimized version of _p_sa() function.
    Converts (nS, nA, nS) matrix to (nS*nA, nS) matrix.

    Args:
        P_sas: (nS, nA, nS) probability matrix
        nS: Number of states
        nA: Number of actions

    Returns:
        P_sa: (nS*nA, nS) probability matrix
    """
    P_sa = np.zeros((nS * nA, nS), dtype=np.float64)

    for s in range(nS):
        for a in range(nA):
            sa = s * nA + a
            P_sa[sa] = P_sas[s, a]

    return P_sa