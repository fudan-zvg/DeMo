import traceback
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import torch
from argoverse.map_representation.map_api import ArgoverseMap


av1_am = ArgoverseMap()
OBJECT_TYPE_MAP = {"AV": 1, "OTHERS": 1, "AGENT": 0}


class Av1Extractor:
    def __init__(
        self,
        radius: float = 50,
        save_path: Path = None,
        mode: str = "train",
    ) -> None:
        self.save_path = save_path
        self.mode = mode
        self.radius = radius

    def save(self, file: Path):
        assert self.save_path is not None

        try:
            data = self.get_data(file)
        except Exception:
            print(traceback.format_exc())
            print("found error while extracting data from {}".format(file))
        save_file = self.save_path / (file.stem + ".pt")
        torch.save(data, save_file)

    def get_data(self, file: Path):
        return self.process(file)
    
    def process(self, raw_path: str, agent_id=None):
        df = pd.read_csv(raw_path)
        
        city = df["CITY_NAME"].values[0]

        timestamps = list(np.sort(df["TIMESTAMP"].unique()))
        actor_ids = list(df["TRACK_ID"].unique())
        num_nodes = len(actor_ids)

        x = torch.zeros(num_nodes, 50, 2, dtype=torch.float)
        x_attr = torch.zeros(num_nodes, 1, dtype=torch.uint8)
        padding_mask = torch.ones(num_nodes, 50, dtype=torch.bool)

        for actor_id, actor_df in df.groupby("TRACK_ID"):
            node_idx = actor_ids.index(actor_id)
            node_steps = [timestamps.index(ts) for ts in actor_df["TIMESTAMP"]]
            object_type = OBJECT_TYPE_MAP[actor_df["OBJECT_TYPE"].values[0]]
            x_attr[node_idx, 0] = object_type
            if object_type == 0:
                agent_idx = node_idx

            padding_mask[node_idx, node_steps] = False

            pos_xy = torch.from_numpy(
                np.stack(
                    [actor_df["X"].values, actor_df["Y"].values],
                    axis=-1,
                )
            ).float()

            x[node_idx, node_steps, :2] = pos_xy

        lane_positions = self.get_lane_features(av1_am, city, x[agent_idx, [0, 9, 19]])
        x = torch.cat([x[[agent_idx]], x[x_attr[:, 0] == 1]])
        padding_mask = torch.cat(
            [padding_mask[[agent_idx]], padding_mask[x_attr[:, 0] == 1]])

        return {
            "x_positions": x,
            "x_valid_mask": ~padding_mask,
            "lane_positions": lane_positions,
            "agent_idx": agent_idx,
            "city": city,
        }

    @staticmethod
    def get_lane_features(
        am, city, positions
    ):
        lane_ids = set()
        for pos in positions:
            lane_ids.update(am.get_lane_ids_in_xy_bbox(pos[0], pos[1], city, 200))

        lane_positions = []

        for lane_id in lane_ids:
            lane_centerline = torch.from_numpy(
                am.get_lane_segment_centerline(lane_id, city)[:, : 2]).float()
            if lane_centerline.shape[0] < 10:
                continue
            lane_positions.append(lane_centerline)

        lane_positions = torch.stack(lane_positions)

        return lane_positions
    