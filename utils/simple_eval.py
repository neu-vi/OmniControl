import numpy as np

# from os.path import join as pjoin
# data_root = './dataset/HumanML3D'
# raw_mean = np.load(pjoin(data_root, 'Mean_raw.npy'))
# raw_std = np.load(pjoin(data_root, 'Std_raw.npy'))

def simple_eval(motion, hint, n_joints=22):
    # mask = hint != 0
    # hint = hint * raw_std + raw_mean
    # hint = hint * mask
    hint = hint.transpose((0, 2, 1))
    hint = hint.reshape(hint.shape[0], n_joints, 3, -1)
    mask = hint.sum(axis=2, keepdims=True) != 0
    loss = np.linalg.norm((motion - hint) * mask, axis=2)
    loss = loss.sum() / mask.sum()
    return loss