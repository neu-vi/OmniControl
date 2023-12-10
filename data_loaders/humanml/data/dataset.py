import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import spacy

from torch.utils.data._utils.collate import default_collate
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from data_loaders.humanml.utils.get_opt import get_opt
from ..scripts.motion_process import recover_root_rot_pos, recover_from_ric
from data_loaders.humanml.utils.metrics import cross_combination_joints
# import spacy

def collate_fn(batch):
    if batch[0][-1] is None:
        batch = [b[:-1] for b in batch]
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text motion matching model, and evaluations'''
class Text2MotionDatasetV2(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer, mode, control_joint=0, density=100):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.mode = mode
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        self.control_joint = control_joint
        self.density = density

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        if 'HumanML3D' in opt.data_root:
            spatial_norm_path = './dataset/humanml_spatial_norm'
        elif 'KIT' in opt.data_root:
            spatial_norm_path = './dataset/kit_spatial_norm'
        else:
            raise NotImplementedError('unknown dataset')
        self.raw_mean = np.load(pjoin(spatial_norm_path, 'Mean_raw.npy'))
        self.raw_std = np.load(pjoin(spatial_norm_path, 'Std_raw.npy'))
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def random_mask_cross(self, joints, n_joints=22, density=1):
        cross_joints = cross_combination_joints()
        choose = np.random.choice(len(cross_joints), 1).item()
        choose_joint = cross_joints[choose]

        length = joints.shape[0]
        choose_seq_num = np.random.choice(length - 1, 1) + 1
        density = self.density
        if density in [1, 2, 5]:
            choose_seq_num = density
        else:
            choose_seq_num = int(length * density / 100)
        choose_seq = np.random.choice(length, choose_seq_num, replace=False)
        choose_seq.sort()
        mask_seq = np.zeros((length, n_joints, 3)).astype(np.bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints
    
    def random_mask(self, joints, n_joints=22, density=1):
        if n_joints == 22:
            # humanml3d
            controllable_joints = np.array([0, 10, 11, 15, 20, 21])
        else:
            # kit
            {1:'root', 2:'BP', 3:'BT', 4:'BLN', 5:'BUN', 6:'LS', 7:'LE', 8:'LW', 9:'RS', 10:'RE', 11:'RW', 12:'LH', 13:'LK', 14:'LA', 15:'LMrot', 16:'LF', 17:'RH', 18:'RK', 19:'RA', 20:'RMrot', 21:'RF'}
            choose_one = ['root', 'BUN', 'LW', 'RW', 'LF', 'RF']
            controllable_joints = np.array([0, 4, 7, 10, 15, 20])

        choose_joint = [self.control_joint]

        length = joints.shape[0]
        choose_seq_num = np.random.choice(length - 1, 1) + 1
        # density = 100
        density = self.density
        if density in [1, 2, 5]:
            choose_seq_num = density
        else:
            choose_seq_num = int(length * density / 100)
        choose_seq = np.random.choice(length, choose_seq_num, replace=False)
        choose_seq.sort()
        mask_seq = np.zeros((length, n_joints, 3)).astype(np.bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints

    def random_mask_train(self, joints, n_joints=22):
        if n_joints == 22:
            controllable_joints = np.array([0, 10, 11, 15, 20, 21])
        else:
            {1:'root', 2:'BP', 3:'BT', 4:'BLN', 5:'BUN', 6:'LS', 7:'LE', 8:'LW', 9:'RS', 10:'RE', 11:'RW', 12:'LH', 13:'LK', 14:'LA', 15:'LMrot', 16:'LF', 17:'RH', 18:'RK', 19:'RA', 20:'RMrot', 21:'RF'}
            choose_one = ['root', 'BUN', 'LW', 'RW', 'LF', 'RF']
            controllable_joints = np.array([0, 4, 7, 10, 15, 20])
        num_joints = len(controllable_joints)
        # joints: length, 22, 3
        num_joints_control = np.random.choice(num_joints, 1)
        # only use one joint during training
        num_joints_control = 1
        choose_joint = np.random.choice(num_joints, num_joints_control, replace=False)
        choose_joint = controllable_joints[choose_joint]

        length = joints.shape[0]
        choose_seq_num = np.random.choice(length - 1, 1) + 1
        choose_seq = np.random.choice(length, choose_seq_num, replace=False)
        choose_seq.sort()
        mask_seq = np.zeros((length, n_joints, 3)).astype(np.bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints

    def random_mask_train_cross(self, joints, n_joints=22):
        from data_loaders.humanml.utils.metrics import cross_combination_joints
        cross_joints = cross_combination_joints()
        choose = np.random.choice(len(cross_joints), 1).item()
        # choose = -1
        choose_joint = cross_joints[choose]

        length = joints.shape[0]
        choose_seq_num = np.random.choice(length - 1, 1) + 1
        choose_seq = np.random.choice(length, choose_seq_num, replace=False)
        choose_seq.sort()
        mask_seq = np.zeros((length, n_joints, 3)).astype(np.bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints
        
    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        n_joints = 22 if motion.shape[-1] == 263 else 21
        # hint is global position of the controllable joints
        joints = recover_from_ric(torch.from_numpy(motion).float(), n_joints)
        joints = joints.numpy()

        # control any joints at any time
        if self.mode == 'train':
            # hint = self.random_mask_train_cross(joints, n_joints)
            hint = self.random_mask_train(joints, n_joints)
        else:
            # hint = self.random_mask_cross(joints, n_joints)
            hint = self.random_mask(joints, n_joints)

        hint = hint.reshape(hint.shape[0], -1)
        if m_length < self.max_motion_length:
            hint = np.concatenate([hint,
                                   np.zeros((self.max_motion_length - m_length, hint.shape[1]))
                                    ], axis=0)

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), hint


class TextOnlyDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.max_length = 20
        self.pointer = 0
        self.fixed_length = 120


        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'text':[text_dict]}
                                new_name_list.append(new_name)
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'text': text_data}
                    new_name_list.append(name)
            except:
                pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        text_list = data['text']

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        return None, None, caption, None, np.array([0]), self.fixed_length, None, None


# A wrapper class for t2m original dataset for MDM purposes
class HumanML3D(data.Dataset):
    def __init__(self, mode, datapath='./dataset/humanml_opt.txt', split="train", control_joint=0, density=100, **kwargs):
        self.mode = mode
        
        self.dataset_name = 't2m'
        self.dataname = 't2m'

        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = f'.'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        opt = get_opt(dataset_opt_path, device)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './dataset'
        self.opt = opt
        print('Loading dataset %s ...' % opt.dataset_name)

        if mode == 'gt':
            # used by T2M models (including evaluators)
            self.mean = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
        elif mode in ['train', 'eval', 'text_only']:
            # used by our models
            self.mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
            self.std = np.load(pjoin(opt.data_root, 'Std.npy'))

        if mode == 'eval':
            # used by T2M models (including evaluators)
            # this is to translate their norms to ours
            self.mean_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))

        self.split_file = pjoin(opt.data_root, f'{split}.txt')
        if mode == 'text_only':
            self.t2m_dataset = TextOnlyDataset(self.opt, self.mean, self.std, self.split_file)
        else:
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
            self.t2m_dataset = Text2MotionDatasetV2(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer, mode, control_joint, density)
            self.num_actions = 1 # dummy placeholder

        assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()

# A wrapper class for t2m original dataset for MDM purposes
class KIT(HumanML3D):
    def __init__(self, mode, datapath='./dataset/kit_opt.txt', split="train", **kwargs):
        super(KIT, self).__init__(mode, datapath, split, **kwargs)