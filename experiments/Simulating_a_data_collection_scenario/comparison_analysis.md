# 深度对比: `simple_protocol.py` vs `visualize_routes_pretrained.py`

本文档旨在精确、深入地比较 **模拟无人机 (`simple_protocol.py`)** 和 **离线测试脚本 (`visualize_routes_pretrained.py`)** 在使用预训练模型进行路径规划时的具体实现差异。

---

## 核心目标

两个脚本都旨在完成同一件事：给定一组节点坐标，使用预训练的 `AttentionModel` 来计算出一条访问所有节点的最佳路径（TSP）。

-   **`visualize_routes_pretrained.py`**: 作为**基准测试工具**。它在一个干净、独立的环境中加载模型和数据，用于验证模型的性能和正确性。
-   **`simple_protocol.py`**: 作为**模拟执行单元**。它在复杂的 `Gradysim` 模拟环境中运行，处理动态变化的状态，并调用模型来指导无人机的行为。

任何不一致的结果都可能源于这两个环境在实现细节上的差异。

---

## 流程并排对比

### 第1步：模型加载

两个脚本都包含一个专门用于加载模型的函数。

| 特性 | `visualize_routes_pretrained.py` (`load_model_for_inference`) | `simple_protocol.py` (`_load_attention_model`) | 分析 |
| :--- | :--- | :--- | :--- |
| **函数签名** | `(checkpoint_path, device, graph_size=None)` | `(self, checkpoint_path, device)` | `simple_protocol` 缺少 `graph_size` 参数，它假设总能从checkpoint中加载`opts`。 |
| **加载检查点** | `torch_load_cpu(checkpoint_path)` | `torch_load_cpu(checkpoint_path)` | **一致**。 |
| **`opts` 处理** | **非常灵活**。优先从checkpoint加载`opts`。如果失败，会使用默认`opts`并允许通过`graph_size`参数覆盖。 | **不够健壮**。尝试从checkpoint加载`opts`。如果失败，会硬编码一个`graph_size=50`的默认值。 | **潜在问题点**。如果模型checkpoint中没有`opts`，`simple_protocol`会强制使用50个节点的配置，这可能与实际情况不符，导致模型结构不匹配。 |
| **模型初始化** | `AttentionModel(...)` | `AttentionModel(...)` | **一致**。都使用相同的参数结构创建模型。 |
| **状态字典加载** | **非常健壮**。它会尝试多种key（`model_state_dict`, `model`），并能处理checkpoint本身就是state dict的情况。 | **较为简单**。它只检查 `model_state_dict` 和 `model`，不如前者全面。 | **潜在问题点**。如果checkpoint的格式比较特殊，`simple_protocol`可能无法正确提取状态字典。 |
| **DataParallel 处理** | 检查`RuntimeError`，然后移除`module.`前缀。 | 检查`RuntimeError`，然后移除`module.`前缀。 | **一致**。 |
| **最终状态** | `model.eval()` | `model.eval()` | **一致**。 |

**模型加载小结**: `visualize_routes_pretrained.py` 中的加载逻辑**更健壮、更灵活**，特别是在处理 `opts` 和不同格式的checkpoint方面。`simple_protocol.py` 的实现在某些边缘情况下可能会失败或加载错误的模型配置。

---

### 第2步：数据准备与模型输入

这是最关键的差异所在。

| 特性 | `visualize_routes_pretrained.py` (`generate_and_plot`) | `simple_protocol.py` (`_solve_tsp_with_attention`) | 分析 |
| :--- | :--- | :--- | :--- |
| **原始数据源** | 从`.pkl`文件加载，例如`dataset[i]`。这是一个包含**所有**节点坐标的张量。 | 从`self.tsp_instances`和`self.sensor_coords_map_tsp_id_key`获取。这是一个**当前阶段需要访问**的节点列表。 | 数据来源不同，但最终都应为节点坐标列表。 |
| **输入构造** | 1. 将数据转为`torch.Tensor`。 2. **增加批处理维度** (`unsqueeze(0)`). | 1. 将`Position3D`列表转为Numpy数组（**只取X,Y坐标**）。 2. 转为`torch.Tensor`。 3. **增加批处理维度** (`unsqueeze(0)`). | **一致**。两者都创建了一个 `[1, num_nodes, 2]` 形状的张量作为模型输入。 |
| **解码类型** | `model.set_decode_type("greedy")` | `self.model.set_decode_type("greedy")` | **一致**。都使用贪心解码，确保结果的确定性。 |
| **模型调用** | `model(instance_data, return_pi=True)` | `self.model(input_tensor, return_pi=True)` | **一致**。 |

**数据准备小结**: 两个脚本在准备模型输入张量的**核心逻辑上是完全一致的**。只要输入的节点坐标列表相同，模型接收到的输入张量也应该是相同的。

---

### 第3步：路径处理（模型输出之后）

| 特性 | `visualize_routes_pretrained.py` | `simple_protocol.py` | 分析 |
| :--- | :--- | :--- | :--- |
| **输出处理** | 直接使用模型输出的`tour_indices`来重排原始节点列表。 | 1. 使用`tour_indices`重排节点列表。 2. **执行一个非常关键的“滚动”操作**。 | **重大差异！** |
| **“滚动”操作** | 无。 | `rolled_tour = ordered_locations[start_node_pos_in_tour:] + ordered_locations[:start_node_pos_in_tour]` | 这是为了让路径从无人机当前位置的**最近节点**开始。 |
| **最终路径** | 完整的闭环路径。 | `rolled_tour[1:]`。返回的是**从第二个节点开始的路径**。 | `simple_protocol`返回的是一个开放路径，因为它假设无人机已经位于第一个节点。 |

**路径处理小结**: `simple_protocol.py` 对模型输出的路径进行了**额外的后处理**。它将路径旋转，使其从特定的起始节点开始，并且**移除了第一个节点**。这是两者路径输出不一致的**最直接和最主要的原因**。

---

## 结论与最终诊断

1.  **主要原因**: `simple_protocol.py` 中的**路径“滚动”和“截断”操作**是导致其输出与 `visualize_routes_pretrained.py` 不一致的根本原因。`visualize_routes_pretrained.py` 显示的是模型原始的、完整的闭环路径，而 `simple_protocol.py` 显示的是经过调整以适应模拟器状态的、不完整的路径。

2.  **次要（但潜在）原因**: `simple_protocol.py` 中**模型加载逻辑不够健壮**。如果使用的模型checkpoint不包含`opts`，或者状态字典的key不是`model`或`model_state_dict`，`simple_protocol`可能会加载一个错误的、不匹配的模型，从而产生完全不同的路径。

**要验证这一点，可以进行以下实验：**

*   **修改 `simple_protocol.py`**: 暂时注释掉 `_solve_tsp_with_attention` 函数中从 `start_node_pos_in_tour` 开始的所有行，直接返回 `ordered_locations`。然后重新运行模拟。如果此时生成的 `tour_stage_*.csv` 文件与 `visualize_routes_pretrained.py` 的输出一致，那么问题就定位在了路径后处理上。
---

## 第4步：多阶段任务执行逻辑 (`simple_protocol.py` 独有)

`visualize_routes_pretrained.py` 的设计目标是处理单个TSP实例。与之相比，`simple_protocol.py` 实现了一套复杂的多阶段任务逻辑，使其能够按顺序处理多个独立的TSP实例。这是两者在宏观行为上的最大区别。

| 行为阶段 | `simple_protocol.py` (`_plan_and_execute_stage`) | `visualize_routes_pretrained.py` | 分析 |
| :--- | :--- | :--- | :--- |
| **任务启动** | 从基站位置 (`self.ground_station_physical_pos`) 开始。 | 不适用。脚本仅处理节点数据，无“启动”概念。 | `simple_protocol` 模拟了完整的任务生命周期。 |
| **实例切换** | 维护一个 `self.current_instance_index` 来追踪当前处理到第几个实例。 | 不适用。 | `simple_protocol` 具备状态管理能力。 |
| **阶段入口点选择** | 1. **第一阶段**: 从基站位置找到第一个实例中最近的节点作为入口点A。 2. **后续阶段**: 从前一阶段的入口点A出发，寻找当前实例中最近的节点作为新的入口点B。 | 不适用。 | **核心差异**。这种“接力式”的入口点选择是多阶段任务的关键。 |
| **阶段内路径规划** | 1. 将入口点A作为TSP计算的起点和终点。 2. 模型计算出遍历路径。 3. 确保最终生成的任务队列会返回到入口点A。 | 仅计算给定节点集的闭环路径。 | `simple_protocol` 的路径规划服务于其阶段性任务目标。 |
| **任务结束** | 当所有实例 (`self.tsp_instances`) 都处理完毕后，生成返回基站的最终路径。 | 不适用。 | `simple_protocol` 确保了任务的闭环。 |

**多阶段逻辑小结**:

`simple_protocol.py` 中的 `_plan_and_execute_stage` 方法是整个多阶段任务的核心。它通过迭代 `self.tsp_instances` 列表，并巧妙地利用前一阶段的终点（即入口点）作为下一阶段的起点，实现了无人机在不同任务区域之间的无缝衔接。

这种设计的行为模式可以概括为：
**基站 -> 实例1入口A -> (遍历实例1) -> 实例1入口A -> 实例2入口B -> (遍历实例2) -> 实例2入口B -> ... -> 返回基站。**

这个逻辑在 `visualize_routes_pretrained.py` 中完全不存在，后者仅仅是一个用于验证单个TSP实例求解能力的工具。