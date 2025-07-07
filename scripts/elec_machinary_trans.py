def oneWaist_2_threeWaist(target):
    for i in range(17, 21):
        target[i] = target[i + 1].copy()

    for i in range(20, 12, -1):
        target[i + 2] = target[i].copy()

    target[13] = 0
    target[14] = 0
    
def threeWaist_2_oneWaist(target):
    for i in range(13, 21):
        target[i] = target[i + 2].copy()
    
    for i in range(20, 16, -1):
        target[i + 1] = target[i].copy()

    target[17] = 0
    target[22] = 0