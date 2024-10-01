import traceback
from pathlib import Path
from typing import List
import numpy as np
import torch
import av2.geometry.interpolate as interp_utils
from av2.map.map_api import ArgoverseStaticMap
from .av2_data_utils import (
    OBJECT_TYPE_MAP,
    OBJECT_TYPE_MAP_COMBINED,
    LaneTypeMap,
    load_av2_df,
)


class Av2Extractor:
    def __init__(
        self,
        radius: float = 150,
        save_path: Path = None,
        mode: str = "train",
        ignore_type: List[int] = [5, 6, 7, 8, 9],
        remove_outlier_actors: bool = True,
    ) -> None:
        self.save_path = save_path
        self.mode = mode
        self.radius = radius
        self.remove_outlier_actors = remove_outlier_actors
        self.ignore_type = ignore_type

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
        df, am, scenario_id = load_av2_df(raw_path)
        city = df.city.values[0]

        timestamps = list(np.sort(df["timestep"].unique()))
        cur_df = df[df["timestep"] == timestamps[49]]
        actor_ids = list(df["track_id"].unique())
        num_nodes = len(actor_ids)

        x = torch.zeros(num_nodes, 110, 2, dtype=torch.float)
        x_attr = torch.zeros(num_nodes, 3, dtype=torch.uint8)
        x_heading = torch.zeros(num_nodes, 110, dtype=torch.float)
        x_velocity = torch.zeros(num_nodes, 110, dtype=torch.float)
        padding_mask = torch.ones(num_nodes, 110, dtype=torch.bool)
        scored_idx = []

        for actor_id, actor_df in df.groupby("track_id"):
            node_idx = actor_ids.index(actor_id)
            node_steps = [timestamps.index(ts) for ts in actor_df["timestep"]]
            object_type = OBJECT_TYPE_MAP[actor_df["object_type"].values[0]]
            object_category = actor_df["object_category"].values[0]
            x_attr[node_idx, 0] = object_type
            x_attr[node_idx, 1] = object_category
            x_attr[node_idx, 2] = OBJECT_TYPE_MAP_COMBINED[
                actor_df["object_type"].values[0]
            ]
            if object_category == 3:
                focal_idx = node_idx
            if object_category == 2:
                scored_idx.append(node_idx)

            padding_mask[node_idx, node_steps] = False

            pos_xy = torch.from_numpy(
                np.stack(
                    [actor_df["position_x"].values, actor_df["position_y"].values],
                    axis=-1,
                )
            ).float()
            heading = torch.from_numpy(actor_df["heading"].values).float()
            velocity = torch.from_numpy(
                actor_df[["velocity_x", "velocity_y"]].values
            ).float()
            velocity_norm = torch.norm(velocity, dim=1)

            x[node_idx, node_steps, :2] = pos_xy
            x_heading[node_idx, node_steps] = heading
            x_velocity[node_idx, node_steps] = velocity_norm

        (
            lane_positions,
            is_intersections,
            lane_attr,
        ) = self.get_lane_features(am)

        return {
            "x_positions": x,
            "x_attr": x_attr,
            "x_angles": x_heading,
            "x_velocity": x_velocity,
            "x_valid_mask": ~padding_mask,
            "lane_positions": lane_positions,
            "lane_attr": lane_attr,
            "is_intersections": is_intersections,
            "scenario_id": scenario_id,
            "agent_ids": actor_ids,
            "focal_idx": focal_idx,
            "scored_idx": scored_idx,
            "city": city,
        }

    @staticmethod
    def get_lane_features(
        am: ArgoverseStaticMap,
    ):
        lane_segments = am.get_scenario_lane_segments()

        lane_positions, is_intersections, lane_attrs = [], [], []
        for segment in lane_segments:
            lane_centerline, lane_width = interp_utils.compute_midpoint_line(
                left_ln_boundary=segment.left_lane_boundary.xyz,
                right_ln_boundary=segment.right_lane_boundary.xyz,
                num_interp_pts=20,
            )
            lane_centerline = torch.from_numpy(lane_centerline[:, :2]).float()
            is_intersection = am.lane_is_in_intersection(segment.id)

            lane_positions.append(lane_centerline)
            is_intersections.append(is_intersection)

            lane_type = LaneTypeMap[segment.lane_type]
            attribute = torch.tensor(
                [lane_type, lane_width, is_intersection], dtype=torch.float
            )
            lane_attrs.append(attribute)

        lane_positions = torch.stack(lane_positions)
        is_intersections = torch.Tensor(is_intersections)
        lane_attrs = torch.stack(lane_attrs, dim=0)

        return (
            lane_positions,
            is_intersections,
            lane_attrs,
        )
