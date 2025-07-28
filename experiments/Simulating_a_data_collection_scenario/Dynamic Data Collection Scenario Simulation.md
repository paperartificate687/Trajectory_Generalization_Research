# 多阶段TSP数据收集场景模拟详解

本文档详细描述了一个基于无人机 (UAV) 的多阶段数据收集模拟场景。在此场景中，UAV按顺序处理一系列独立的旅行商问题 (TSP) 实例。内容依据以下核心代码文件：

- 主程序: [`showcases/Simulating_a_data_collection_scenario/main.py`](showcases/Simulating_a_data_collection_scenario/main.py)
- 协议定义: [`showcases/Simulating_a_data_collection_scenario/simple_protocol.py`](showcases/Simulating_a_data_collection_scenario/simple_protocol.py)
- 轨迹绘图: [`showcases/Simulating_a_data_collection_scenario/plot_trajectories.py`](showcases/Simulating_a_data_collection_scenario/plot_trajectories.py)

## 1. 场景概述

模拟的核心是一个无人机 (UAV) 在一个广阔区域内执行多阶段的数据收集任务。整个任务由多个独立的传感器集群（TSP实例）组成。UAV的目标是按顺序访问每个集群，激活其中的所有传感器，高效地规划路径收集它们的数据，然后在完成一个集群后继续下一个，直到所有集群的任务完成，最终返回地面站。

### 1.1. 节点类型与行为

模拟包含三种主要类型的节点：

#### 1.1.1. 传感器节点 (Sensor Nodes)

- **协议类:** [`SimpleSensorProtocol`](showcases/Simulating_a_data_collection_scenario/simple_protocol.py:100)
- **位置:** 所有传感器的位置从一个包含多个TSP实例的Pickle文件 (`.pkl`) 中加载，例如 `data/tsp/tsp50_train_seed1111_size10K.pkl`。`main.py` 会读取这些实例并将其坐标映射到模拟环境中。
- **状态与行为:** 传感器状态由 [`SensorStatus`](showcases/Simulating_a_data_collection_scenario/simple_protocol.py:74) 枚举定义，并通过调用 [`_update_sensor_status_file`](showcases/Simulating_a_data_collection_scenario/simple_protocol.py:31) 实时写入 `sensor_statuses.json`。
  * **原始 (RAW):** [`SensorStatus.RAW`](showcases/Simulating_a_data_collection_scenario/simple_protocol.py:75)。所有传感器在 `main.py` 初始化状态文件时的初始值。
  * **待机 (Standby):** [`SensorStatus.STANDBY`](showcases/Simulating_a_data_collection_scenario/simple_protocol.py:76)。在模拟开始前，所有传感器都被设置为此状态。它们不产生数据也不请求服务。
  * **激活 (Active):** [`SensorStatus.ACTIVE`](showcases/Simulating_a_data_collection_scenario/simple_protocol.py:77)。当UAV开始处理包含该传感器的TSP实例时，UAV会广播一条消息，该传感器接收到后将调用其 `activate()` 方法 ([`showcases/Simulating_a_data_collection_scenario/simple_protocol.py:202`](showcases/Simulating_a_data_collection_scenario/simple_protocol.py:202))。激活后，传感器开始生成数据包。
  * **已服务 (Serviced):** [`SensorStatus.SERVICED`](showcases/Simulating_a_data_collection_scenario/simple_protocol.py:78)。当UAV进入其通信范围内时，传感器将数据发送给UAV，状态变为“已服务”，并停止活动。

#### 1.1.2. 无人机 (UAV)

- **协议类:** [`SimpleUAVProtocol`](showcases/Simulating_a_data_collection_scenario/simple_protocol.py:314)
- **初始化:**
  * 在 `main.py` 中创建，并通过 `configure` 方法 ([`showcases/Simulating_a_data_collection_scenario/simple_protocol.py:327`](showcases/Simulating_a_data_collection_scenario/simple_protocol.py:327)) 配置参数，最核心的是 `tsp_instances`，这是一个包含多个列表的列表，每个子列表代表一个独立的TSP实例，其中包含该实例所有传感器的TSP ID。
- **核心任务逻辑 (多阶段任务):**
  * **启动任务:** 初始化后，UAV调用 `_start_mission` ([`showcases/Simulating_a_data_collection_scenario/simple_protocol.py:389`](showcases/Simulating_a_data_collection_scenario/simple_protocol.py:389))，开始处理第一个TSP实例。
  * **阶段规划与执行 (`_plan_and_execute_stage`):** ([`showcases/Simulating_a_data_collection_scenario/simple_protocol.py:494`](showcases/Simulating_a_data_collection_scenario/simple_protocol.py:494))
    1.  **检查任务:** 检查是否还有未处理的TSP实例。如果没有，则规划返回地面站的最终路径。
    2.  **激活当前阶段传感器:** UAV广播一条消息，其中包含当前TSP实例中所有传感器的ID，以激活它们。
    3.  **构建并求解闭环TSP:**
        *   **目标节点:** UAV的当前位置（作为起点/终点）和当前实例中所有传感器的位置。
        *   **求解:** 调用 `_solve_tsp_with_lkh` ([`showcases/Simulating_a_data_collection_scenario/simple_protocol.py:430`](showcases/Simulating_a_data_collection_scenario/simple_protocol.py:430))，使用LKH-3算法求解一个标准的闭环TSP问题，计算出访问当前阶段所有传感器的最优路径。
    4.  **执行路径:** UAV按照计算出的路径飞行，并收集沿途传感器的数据。
  * **处理下一阶段:** 当一个阶段的所有目标点都访问完毕后，UAV会自动调用 `_plan_and_execute_stage` 进入下一个阶段，重复上述过程。
  * **任务完成:** 所有TSP实例处理完毕后，UAV返回地面站，转储所有收集到的数据。
- **日志记录:** 在 `finish` 方法 ([`showcases/Simulating_a_data_collection_scenario/simple_protocol.py:597`](showcases/Simulating_a_data_collection_scenario/simple_protocol.py:597)) 中，将整个任务的飞行轨迹、能量消耗等信息保存到文件中。

#### 1.1.3. 地面站 (Ground Station)

- **协议类:** [`SimpleGroundStationProtocol`](showcases/Simulating_a_data_collection_scenario/simple_protocol.py:649)
- **行为:** 接收UAV最终转储的数据，并发送ACK确认消息以结束仿真。

## 2. 程序执行逻辑

### 2.1. `run_simulation` 函数

1.  **加载TSP实例:** 从指定的 `.pkl` 文件加载多个TSP实例的坐标数据。
2.  **初始化节点:** 创建地面站、UAV和所有TSP实例中定义的传感器节点。
3.  **配置UAV:** 将TSP实例列表和所有传感器的坐标图传递给UAV。
4.  **设置传感器初始状态:** 将所有传感器设置为 `STANDBY` 状态。
5.  **启动模拟:** 调用 `simulation.start_simulation()`。
6.  **结果分类:** 仿真结束后，读取最终的 `sensor_statuses.json`，并根据传感器的最终状态（如 `SERVICED`, `STANDBY` 等）为其分配一个绘图类别，结果保存到 `plot_categories.json`。

### 2.2. `main` 函数

定义实验参数（如要加载多少个TSP实例），循环调用 `run_simulation`，并最终将所有运行的性能指标汇总保存到 `simulation_results.csv`。

## 3. 结果可视化

通过运行 [`plot_trajectories.py`](showcases/Simulating_a_data_collection_scenario/plot_trajectories.py) 脚本对模拟结果进行可视化。

- **输入:**
  * `sensor_locations_forest.json`: **[注意]** 脚本当前存在一个不一致问题，它从此文件加载传感器坐标，而主程序 `main.py` 是从 `.pkl` 文件加载。为保证绘图正确，需要确保此JSON文件包含与模拟中使用的数据相匹配的坐标。
  * `plot_categories.json`: `main.py` 生成的传感器分类文件。
  * `uav_*_trajectory.csv`: UAV的完整飞行轨迹文件。
- **输出:** 生成一张名为 `trajectories.png` 的3D散点图，其中包含：
  * **地面站:** 黑色金字塔标记。
  * **传感器节点 (根据类别):**
    * `Initial Active Sensors`: 绿色圆圈，通常代表在第一个TSP实例中被激活并服务的传感器。
    * `Dynamic Newly Activated`: 青色'P'标记，代表在后续TSP实例中被激活并服务的传感器。
    * `Activated -> Standby (Unserviced)`: 橙色小'x'，表示被激活但最终未被服务的传感器（例如，模拟提前结束）。
    * `Other Standby Sensors`: 灰色方块（默认不绘制），表示从未被激活的传感器。
  * **UAV 轨迹:**
    * `UAV ... Trajectory`: 一条连续的彩色线条，表示UAV完成所有阶段任务的完整路径。

## 4. 输出文件

- `showcases/Simulating_a_data_collection_scenario/simulation_results.csv`: 所有模拟运行的详细性能指标。
- `showcases/Simulating_a_data_collection_scenario/sensor_statuses.json`: 单次运行中，传感器状态的实时记录。
- `showcases/Simulating_a_data_collection_scenario/plot_categories.json`: 为绘图脚本准备的传感器分类数据。
- `showcases/Simulating_a_data_collection_scenario/uav_{id}_trajectory.csv`: UAV完成所有任务的完整飞行轨迹。
- `showcases/Simulating_a_data_collection_scenario/uav_{id}_energy.csv`: UAV的能量消耗记录。
- `gs_{id}_received_data.json`: 地面站接收到的原始数据批次。
- `trajectories.png`: 最终生成的可视化轨迹图。
