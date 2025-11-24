import numpy as np

def p_sas(P, nS, nA):
    """
    Build transition probability matrix P[s,a,s'] from dict format.
    Optimized version - ~50-100x faster than original.
    """
    P_sas = np.zeros((nS, nA, nS), dtype=np.float32)
    
    for s in range(nS):
        for a in range(nA):
            transitions = P[s][a]
            # Direct assignment - no nested loop over s1!
            for prob, s_next, _, _ in transitions:
                if prob > 0:  # Skip zero probabilities
                    P_sas[s, a, s_next] += prob
    
    return P_sas


def r_sas(P, nS, nA):
    """
    Build reward matrix R[s,a,s'] from dict format.
    Optimized version.
    """
    R_sas = np.zeros((nS, nA, nS), dtype=np.float32)
    
    for s in range(nS):
        for a in range(nA):
            transitions = P[s][a]
            for prob, s_next, reward, _ in transitions:
                if prob > 0:  # Only set reward if transition exists
                    R_sas[s, a, s_next] = reward
    
    return R_sas


def p_sa(P_sas, nS, nA):
    """
    Reshape P[s,a,s'] to P[sa,s'] where sa is flattened (s,a) index.
    Vectorized version - instant instead of loops.
    """
    # Original: loops through everything
    # Optimized: just reshape!
    return P_sas.reshape(nS * nA, nS)


def r_sa(R_sas, nS, nA):
    """
    Reshape R[s,a,s'] to R[sa,s'] where sa is flattened (s,a) index.
    Vectorized version - instant instead of loops.
    """
    return R_sas.reshape(nS * nA, nS)