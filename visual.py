from pathlib import Path
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf
import os 
import torch
from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap
from src.datamodule.av2_dataset import Av2Dataset
from src.datamodule.av2_dataset import collate_fn
from src.utils.vis import visualize_scenario
import hydra
from hydra import compose, initialize_config_dir
from pathlib import Path as P
from tqdm import tqdm


def move_to_cuda(dictionary, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            dictionary[key] = value.to(device)
        elif isinstance(value, dict):
            move_to_cuda(value, device)
    return dictionary

# 各种配置
data_root_pro = Path("data/DeMo_processed") 
data_root = Path("data/val")
config_path = P("conf").absolute()  # 使用绝对路径
ckpt = "ckpts/DeMo.ckpt"
time_steps = [30, 40, 50]

# 设定范围
start_num = 1
end_num = 2000

dataset = Av2Dataset(data_root=data_root_pro, split='val', 
                     num_historical_steps=30, sequence_origins=[30, 40, 50], 
                     radius=150.0, train_mode='only_focal')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用 hydra 方式加载配置（与 train.py 一致）
with initialize_config_dir(version_base=None, config_dir=str(config_path)):
    cfg = compose(config_name="config")

# 使用 hydra instantiate 创建模型（与 train.py 一致）
model = instantiate(cfg.model.target)

# 加载 checkpoint 权重
checkpoint = torch.load(ckpt, map_location=device)
model.load_state_dict(checkpoint['state_dict'])

model = model.eval()
model = model.to(device)

# 确保 end_num 不超过数据集长度
end_num = min(end_num, len(dataset))

# 批量生成可视化
for num in tqdm(range(start_num, end_num + 1), desc="Generating visualizations"):
    try:
        path_num = f'outputs/outfig/{num}'
        os.makedirs(path_num, exist_ok=True)

        data = dataset[num]

        scene_id = data[0]["scenario_id"]
        scene_file = data_root / scene_id / ("scenario_" + scene_id + ".parquet")
        map_file = data_root / scene_id / ("log_map_archive_" + scene_id + ".json")
        
        # 检查文件是否存在
        if not scene_file.exists() or not map_file.exists():
            print(f"Skipping scene {num}: files not found")
            continue
            
        scenario = scenario_serialization.load_argoverse_scenario_parquet(scene_file)
        static_map = ArgoverseStaticMap.from_json(map_file)

        for da in data:
            da = move_to_cuda(da)

        with torch.no_grad():
            predictions, probs = model.predict(collate_fn([data]))

        for i, time_step, prediction, prob in zip(range(len(predictions)), time_steps, predictions, probs):
            visualize_scenario(scenario, static_map, prediction=prediction.squeeze(0), timestep=time_step, save_path=f'{path_num}/outfigure{num}_{i}')
    
    except Exception as e:
        print(f"Error processing scene {num}: {e}")
        continue

print(f"Visualization generation completed! Range: {start_num} - {end_num}")
