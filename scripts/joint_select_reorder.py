import numpy as np

def extend_joint(action_3waist):
    action_1waist = np.zeros(23)
    
    action_1waist[0] = action_3waist[0]
    action_1waist[1] = action_3waist[1]
    action_1waist[2] = action_3waist[2]
    action_1waist[3] = action_3waist[3]
    action_1waist[4] = action_3waist[4]
    action_1waist[5] = action_3waist[5]

    action_1waist[6] = action_3waist[6]
    action_1waist[7] = action_3waist[7]
    action_1waist[8] = action_3waist[8]
    action_1waist[9] = action_3waist[9]
    action_1waist[10] = action_3waist[10]
    action_1waist[11] = action_3waist[11]

    action_1waist[12] = action_3waist[12]

    action_1waist[13] = action_3waist[15]
    action_1waist[14] = action_3waist[16]
    action_1waist[15] = action_3waist[17]
    action_1waist[16] = action_3waist[18]
    action_1waist[17] = 0.0

    action_1waist[18] = action_3waist[19]
    action_1waist[19] = action_3waist[20]
    action_1waist[20] = action_3waist[21]
    action_1waist[21] = action_3waist[22]
    action_1waist[22] = 0.0

    return action_1waist

def obs_match(obs_1waist):
    obs_3waist = np.zeros(23)
    
    obs_3waist[0] = obs_1waist[0]
    obs_3waist[1] = obs_1waist[1]
    obs_3waist[2] = obs_1waist[2]
    obs_3waist[3] = obs_1waist[3]
    obs_3waist[4] = obs_1waist[4]
    obs_3waist[5] = obs_1waist[5]

    obs_3waist[6] = obs_1waist[6]
    obs_3waist[7] = obs_1waist[7]
    obs_3waist[8] = obs_1waist[8]
    obs_3waist[9] = obs_1waist[9]
    obs_3waist[10] = obs_1waist[10]
    obs_3waist[11] = obs_1waist[11]

    obs_3waist[12] = obs_1waist[12]
    obs_3waist[13] = 0.0
    obs_3waist[14] = 0.0

    obs_3waist[15] = obs_1waist[13]
    obs_3waist[16] = obs_1waist[14]
    obs_3waist[17] = obs_1waist[15]
    obs_3waist[18] = obs_1waist[16]

    obs_3waist[19] = obs_1waist[18]
    obs_3waist[20] = obs_1waist[19]
    obs_3waist[21] = obs_1waist[20]
    obs_3waist[22] = obs_1waist[21]

    return obs_3waist





