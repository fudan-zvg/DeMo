from typing import List
from pathlib import Path
import numpy as np
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class Av1Dataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str = None,
        num_historical_steps: int = 20,
        sequence_origins: List[int] = [20],
        radius: float = 50.0,
    ):
        assert sequence_origins[-1] == 20 and num_historical_steps <= 20
        assert split in ['train', 'val', 'test']
        super(Av1Dataset, self).__init__()
        self.data_folder = Path(data_root) / split
        self.file_list = sorted(list(self.data_folder.glob('*.pt')))
        self.num_historical_steps = num_historical_steps
        # self.num_future_steps = 0 if split =='test' else 30
        self.num_future_steps = 30
        self.sequence_origins = sequence_origins
        self.radius = radius

        print(
            f'data root: {data_root}/{split}, total number of files: {len(self.file_list)}'
        )

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        data = torch.load(self.file_list[index])
        scenario_id = os.path.splitext(os.path.basename(self.file_list[index]))[0]
        data = self.process(data, scenario_id)
        return data
    
    def process(self, data, scenario_id):
        sequence_data = []
        for cur_step in self.sequence_origins:
            ag_dict = self.process_single_agent(data, cur_step)
            ag_dict['scenario_id'] = scenario_id
            sequence_data.append(ag_dict)
        return sequence_data

    def process_single_agent(self, data, step=20):
        origin = data['x_positions'][0, step - 1].double()
        heading_vector = data['x_positions'][0, step - 1] - data['x_positions'][0, 0]
        theta = torch.atan2(heading_vector[1].double(), heading_vector[0].double())
        rotate_mat = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)],
            ],
        )
        ag_mask = torch.norm(data['x_positions'][:, step - 1] - origin, dim=-1) < self.radius
        ag_mask = ag_mask * data['x_valid_mask'][:, step - 1]

        st, ed = step - self.num_historical_steps, step + self.num_future_steps
        pos = data['x_positions'][ag_mask, st: ed]
        valid_mask = data['x_valid_mask'][ag_mask, st: ed]
        pos[valid_mask] = torch.matmul(pos[valid_mask].double() - origin, rotate_mat).to(torch.float32)
        head = pos.new_zeros(pos.shape[0])
        head_mask = valid_mask[:, step - 1] & valid_mask[:, step - 2]
        heading_vector = (pos[:, step - 1] - pos[:, step - 2])[head_mask]
        head[head_mask] = (torch.atan2(heading_vector[:, 1], heading_vector[:, 0]) + np.pi) % (2 * np.pi) - np.pi

        l_pos = data['lane_positions']
        l_pos = torch.matmul(l_pos.reshape(-1, 2).double() - origin, rotate_mat).reshape(-1, l_pos.size(1), 2).to(torch.float32)

        l_ctr = l_pos[:, 4:6].mean(dim=1)
        l_head = torch.atan2(
            l_pos[:, 5, 1] - l_pos[:, 4, 1],
            l_pos[:, 5, 0] - l_pos[:, 4, 0],
        )
        l_valid_mask = (
            (l_pos[:, :, 0] > -self.radius) & (l_pos[:, :, 0] < self.radius)
            & (l_pos[:, :, 1] > -self.radius) & (l_pos[:, :, 1] < self.radius)
        )

        l_mask = l_valid_mask.any(dim=-1)
        l_pos = l_pos[l_mask]
        l_ctr = l_ctr[l_mask]
        l_head = l_head[l_mask]
        l_valid_mask = l_valid_mask[l_mask]

        l_pos = torch.where(
            l_valid_mask[..., None], l_pos, torch.zeros_like(l_pos)
        )

        nearest_dist = torch.cdist(pos[:, self.num_historical_steps - 1, :2],
                                   l_pos.view(-1, 2)).min(dim=1).values
        ag_mask = nearest_dist < 5
        ag_mask[0] = True
        pos = pos[ag_mask]
        head = head[ag_mask]
        valid_mask = valid_mask[ag_mask]

        pos_ctr = pos[:, self.num_historical_steps - 1].clone()
        if self.num_future_steps > 0:
            pos, target = pos[:, :self.num_historical_steps], pos[:, self.num_historical_steps:]
            target_mask = valid_mask[:, [self.num_historical_steps - 1]] & valid_mask[:, self.num_historical_steps:]
            valid_mask = valid_mask[:, :self.num_historical_steps]
            target = torch.where(
                target_mask.unsqueeze(-1),
                target - pos_ctr.unsqueeze(1), torch.zeros(pos_ctr.size(0), 30, 2),   
            )
        else:
            target = target_mask = None

        diff_mask = valid_mask[:, :self.num_historical_steps - 1] & valid_mask[:, 1: self.num_historical_steps]
        tmp_pos = pos.clone()
        pos_diff = pos[:, 1:self.num_historical_steps] - pos[:, :self.num_historical_steps - 1]
        pos[:, 1:self.num_historical_steps] = torch.where(
            diff_mask.unsqueeze(-1),
            pos_diff, torch.zeros(pos.size(0), self.num_historical_steps - 1, 2)
        )
        pos[:, 0] = torch.zeros(pos.size(0), 2)

        return {
            'target': target,
            'target_mask': target_mask,
            'x_positions_diff': pos,
            'x_positions': tmp_pos,
            'x_centers': pos_ctr,
            'x_angles': head,
            'x_valid_mask': valid_mask,
            'lane_positions': l_pos,
            'lane_centers': l_ctr,
            'lane_angles': l_head,
            'lane_valid_mask': l_valid_mask,
            'origin': origin.view(1, 2),
            'theta': theta.view(1),
            'city': data['city'],
            'timestamp': torch.Tensor([step * 0.1])
        }

        
def collate_fn(seq_batch):
    seq_data = []
    for i in range(len(seq_batch[0])):
        batch = [b[i] for b in seq_batch]
        data = {}

        for key in [
            'x_positions_diff',
            'x_positions',
            'x_centers',
            'x_angles',
            'lane_positions',
            'lane_centers',
            'lane_angles',
        ]:
            data[key] = pad_sequence([b[key] for b in batch], batch_first=True)

        if batch[0]['target'] is not None:
            data['target'] = pad_sequence([b['target'] for b in batch], batch_first=True)
            data['target_mask'] = pad_sequence(
                [b['target_mask'] for b in batch], batch_first=True, padding_value=False
            )

        for key in ['x_valid_mask', 'lane_valid_mask']:
            data[key] = pad_sequence(
                [b[key] for b in batch], batch_first=True, padding_value=False
            )

        data['x_key_valid_mask'] = data['x_valid_mask'].any(-1)
        data['lane_key_valid_mask'] = data['lane_valid_mask'].any(-1)

        data['origin'] = torch.cat([b['origin'] for b in batch], dim=0)
        data['theta'] = torch.cat([b['theta'] for b in batch])
        data['scenario_id'] = [b['scenario_id'] for b in batch]
        data['timestamp'] = torch.cat([b['timestamp'] for b in batch])
        seq_data.append(data)
    return seq_data
