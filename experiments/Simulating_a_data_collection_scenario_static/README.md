# 基于Gradysim的ADAPT-GUAV仿真框架

此目录包含一个GrADyS-SIM仿真框架，旨在评估和基准测试无人机数据采集轨迹，特别是用于测试ADAPT-GUAV论文中概述的原则。

## 场景描述

本模拟解决的是**轨迹泛化问题（TGP**），其中无人机必须高效地从网络拓扑可能显著变化（例如，集群城市与稀疏随机）且动态变化（节点加入或离开）的物联网设备中收集数据。

该框架基于ADAPT-GUAV提出的**训练-推理分离**范式：

1. **离线策略训练（外部步骤）**：使用计算密集型算法（如ADAPT-GUAV强化学习模型或传统的LKH求解器）处理一组物联网设备位置（`sensor_locations.tsp`），并生成优化的访问序列（`result_tour.tsp`）。这项繁重的计算是在离线完成的。
2. **在线推理与评估（本模拟）**：本模拟框架加载预先计算的轨迹。无人机作为**推理代理**执行给定路径。模拟环境随后在现实物理和网络模型中对该路径的性能进行严格基准测试，捕捉关键性能指标。

该模拟器的主要目的是回答以下问题：**给定轨迹生成策略在现实世界中的效果如何？**

主要组件包括：

* **传感器节点 (`SimpleSensorProtocol`):** 这些静态节点生成数据包，并等待无人机进入收集范围。它们在 `sensor_locations.tsp` 中定义的空间分布代表了正在测试的特定物联网环境（例如，城市、森林）。当无人机飞到其通信范围内（特别是正上方）时，传感器会将累积的数据包发送给无人机。

- **无人机节点 (`SimpleUAVProtocol`):** 无人机充当执行和推理代理。它从TSP巡游文件中加载预先计算的航线并执行。它会定期广播心跳消息（包含其当前位置），从传感器收集数据包。完成任务后，无人机将所有收集到的数据转储到地面站，然后尝试降落在地面站的位置。无人机还会记录其飞行轨迹和能量消耗到 CSV 文件中。其性能，包括飞行路径、能耗和收集的数据，被详细记录。
- **地面站节点 (`SimpleGroundStationProtocol`):** 地面站充当中央性能评估器。它在无人机任务结束时接收数据转储，发送确认消息，并计算关键的全系统指标，如端到端数据延迟和整体吞吐量，这些对于判断执行轨迹的质量至关重要。

## 文件说明

* [`main.py`](https://www.google.com/search?q=showcases/Simulating_a_data_collection_scenario/main.py:1): 配置和启动 GrADyS-SIM 评估环境的主体脚本。

- [`simple_protocol.py`](https://www.google.com/search?q=showcases/Simulating_a_data_collection_scenario/simple_protocol.py:1): 定义传感器、无人机和地面站的行为。
  - `SimpleUAVProtocol` 是评估的核心，因为它动态加载并执行在指定的 TSP 文件中定义的轨迹。它还记录生成的轨迹和能耗到 `.csv` 文件。
- [`plot_trajectories.py`](https://www.google.com/search?q=showcases/Simulating_a_data_collection_scenario/plot_trajectories.py:1): 一个用于可视化仿真结果的实用脚本，将无人机的飞行路径与传感器和地面站的位置进行绘图。
- `sensor_locations.tsp`：**(实验输入)** 此文件定义了实验中特定物联网网络拓扑。为了测试泛化能力，您可以创建此文件的多个版本来表示不同的部署场景（例如，`sensors_urban.tsp`，`sensors_forest.tsp`）。
- `result_single_100.tsp`（示例）：**(实验输入)** 这是一个 TSP 旅行文件，代表离线轨迹规划器的输出。实验的目标是测试在此定义的旅行路径的质量。您可以生成不同的版本来比较算法（例如，`tour_adapt_guav.tsp`，`tour_lkh.tsp`）。
- `uav_<ID>_trajectory.csv`：**(实验输出)** 无人机随时间变化的 3D 坐标日志，在每个仿真运行后生成。
- `uav_<ID>_energy.csv`：**(实验输出)** 无人机随时间变化的能量水平日志，在每个仿真运行后生成。

## 如何运行

按照以下步骤评估轨迹生成策略（例如，ADAPT-GUAV 与基线）：

1. **定义网络拓扑：**

- 创建或选择一个定义传感器坐标的 `.tsp` 文件（例如，`sensors_urban.tsp`）。这是您的测试环境。

2. **生成轨迹（离线步骤）：**

- 使用您的外部轨迹规划算法（例如，训练好的 ADAPT-GUAV 模型，LKH 求解器）处理步骤 1 中的传感器位置。
- 将生成的访问序列保存为 TSP 游览文件（例如，`tour_urban_adapt_guav.tsp`）。

3. **配置模拟：**

- 打开 [simple_protocol.py](https://www.google.com/search?q=showcases/Simulating_a_data_collection_scenario/simple_protocol.py:1)。
- 在 `SimpleUAVProtocol` 类中，更新文件路径常量以指向您选择的文件：

Python

```

SENSOR_LOCATIONS_TSP_FILE = "showcases/Simulating_a_data_collection_scenario/sensors_urban.tsp"

TSP_TOUR_FILE = "showcases/Simulating_a_data_collection_scenario/tour_urban_adapt_guav.tsp"

```

4. **运行评估：**

- 从您的终端执行主模拟脚本：

Bash

```

python showcases/Simulating_a_data_collection_scenario/main.py

```

- 模拟将运行，无人机将遵循您的游览文件中定义的路径。最后，地面站将记录性能指标。

5. **分析和可视化结果：**

- 检查控制台输出以获取地面站的延迟和吞吐量统计信息。
- 分析生成的 `uav_*_energy.csv` 以计算能源效率。
- 运行绘图脚本以可视化执行的轨迹：

Bash

```

python showcases/Simulating_a_data_collection_scenario/plot_trajectories.py

```

- 通过对同一传感器布局的不同游览文件（来自不同算法）重复此过程，您可以直接比较它们的性能。

## 依赖

* `gradysim`: 仿真框架。
* `matplotlib`: 用于绘图。

可以通过 `requirements.txt` (如果项目根目录提供) 或直接 `pip install matplotlib` 来安装 `matplotlib`。
