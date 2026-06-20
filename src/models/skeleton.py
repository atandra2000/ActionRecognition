"""
Shared skeleton topology definitions for NTU RGB+D format.
"""

# NTU RGB+D skeleton connections (1-indexed joint IDs)
SKELETON_CONNECTIONS = [
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
    (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
    (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
    (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
    (22, 23), (23, 8), (24, 25), (25, 12)
]

# Number of joints for NTU RGB+D
NUM_NTU_JOINTS = 25

# Center joint (spine base) - 1-indexed
CENTER_JOINT = 21  # 1-indexed, becomes 20 in 0-indexed