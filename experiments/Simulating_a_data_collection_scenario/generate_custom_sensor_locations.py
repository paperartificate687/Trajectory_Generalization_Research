import json
import numpy as np
import random

def generate_poisson_process_points(lambda_intensity, area_x, area_y, num_points_to_generate):
    """
    生成泊松过程点。
    注意：真正的泊松过程点数量是随机的。这里我们为了方便控制生成点的数量。
    lambda_intensity 实际上在这里更多地影响点的分散程度，而不是严格意义上的密度。
    """
    points = []
    # 近似地，我们可以认为点在区域内均匀随机分布
    # lambda_intensity 越小，我们假设点越稀疏，因此在生成固定数量点时，它们可能需要更大的有效区域来模拟稀疏性。
    # 这里简化处理，直接在指定区域内随机生成。
    for _ in range(num_points_to_generate):
        x = random.uniform(0, area_x)
        y = random.uniform(0, area_y)
        points.append([x, y, 0.0])
    return points

def generate_gaussian_cluster_points(mu, sigma, num_points_per_cluster, area_x, area_y):
    """
    生成高斯集群点。
    mu: 均值 [mux, muy]
    sigma: 协方差矩阵 [[sig_xx, sig_xy], [sig_yx, sig_yy]] 或标量（各向同性）
    """
    points = []
    if isinstance(sigma, (int, float)): # 各向同性
        cov_matrix = [[sigma**2, 0], [0, sigma**2]]
    else: # 各向异性
        cov_matrix = sigma

    generated_points = np.random.multivariate_normal(mu, cov_matrix, num_points_per_cluster)
    for p in generated_points:
        # 确保点在边界内，如果超出则重新采样或裁剪 (这里简化为裁剪)
        x = max(0, min(p[0], area_x))
        y = max(0, min(p[1], area_y))
        points.append([x, y, 0.0])
    return points

def generate_grid_points(rows, cols, spacing_x, spacing_y, area_x, area_y, jitter=0.0):
    """
    生成网格点。
    jitter: 给网格点增加的随机抖动幅度。
    """
    points = []
    start_x = (area_x - (cols - 1) * spacing_x) / 2
    start_y = (area_y - (rows - 1) * spacing_y) / 2

    for r in range(rows):
        for c in range(cols):
            x = start_x + c * spacing_x + random.uniform(-jitter, jitter)
            y = start_y + r * spacing_y + random.uniform(-jitter, jitter)
            # 确保点在边界内
            x = max(0, min(x, area_x))
            y = max(0, min(y, area_y))
            points.append([x, y, 0.0])
    return points

def format_points_to_json(points, area_x, area_y):
    """
    将点列表格式化为目标 JSON 结构。
    节点 "0" 被强制设置为区域中心作为基站。
    """
    output_dict = {}
    
    # 节点 0 是基站，位于中心
    base_station_pos = [area_x / 2, area_y / 2, 0.0]
    output_dict["0"] = base_station_pos
    
    # 确保总点数正确，从原始列表中移除一个点
    if len(points) > 0:
        points.pop() # 移除最后一个点以腾出空间给基站

    # 其他点从 ID 1 开始
    for i, p in enumerate(points):
        output_dict[str(i + 1)] = [float(p[0]), float(p[1]), float(p[2])]
        
    return output_dict

def save_to_json(data, filename):
    """
    将数据保存到 JSON 文件。
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Generated {filename}")

# --- 定义场景参数 ---
AREA_X = 500.0  # 假设的区域宽度
AREA_Y = 500.0  # 假设的区域高度
TOTAL_POINTS = 1000 # 每个场景生成的总点数 (可以根据需要调整)

# 1. 稀疏随机分布 (森林)
# alpha = 0, D(x,y) = P_lambda(x,y), low lambda <= 0.1
def generate_forest_scenario():
    alpha = 0.0
    # lambda_intensity 理论上影响点密度，但我们固定总点数，所以它更多是概念上的
    # 对于固定点数，我们可以直接生成随机点
    points = generate_poisson_process_points(lambda_intensity=0.1, area_x=AREA_X, area_y=AREA_Y, num_points_to_generate=TOTAL_POINTS)
    return format_points_to_json(points, AREA_X, AREA_Y)

# 2. 密集集群分布 (城市)
# alpha -> 0.8. D(x,y) = 0.2*P_lambda(x,y) + 0.8*sum(1/C * N(mu_c, sigma^2*I))
def generate_urban_scenario():
    alpha = 0.8
    num_random_points = int((1 - alpha) * TOTAL_POINTS)
    num_cluster_points_total = TOTAL_POINTS - num_random_points

    points = generate_poisson_process_points(lambda_intensity=0.5, area_x=AREA_X, area_y=AREA_Y, num_points_to_generate=num_random_points)

    num_clusters = 5  # 假设 C=5 个集群
    if num_clusters == 0: # 避免除以零
        points.extend(generate_poisson_process_points(lambda_intensity=0.5, area_x=AREA_X, area_y=AREA_Y, num_points_to_generate=num_cluster_points_total))
    else:
        points_per_cluster_ideal = num_cluster_points_total // num_clusters
        
        # 确保总点数正确分配
        assigned_cluster_points = 0
        cluster_points_list = []

        for i in range(num_clusters):
            # 随机选择集群中心
            mu_c = [random.uniform(0.1 * AREA_X, 0.9 * AREA_X), random.uniform(0.1 * AREA_Y, 0.9 * AREA_Y)]
            sigma_c = random.uniform(15, 40) # 集群的扩展范围

            # 分配点数，确保最后一个集群补齐差额
            if i < num_clusters -1:
                num_points_for_this_cluster = points_per_cluster_ideal
            else:
                num_points_for_this_cluster = num_cluster_points_total - assigned_cluster_points
            
            if num_points_for_this_cluster > 0:
                cluster_points_list.extend(generate_gaussian_cluster_points(mu_c, sigma_c, num_points_for_this_cluster, AREA_X, AREA_Y))
                assigned_cluster_points += num_points_for_this_cluster
        
        points.extend(cluster_points_list)
    
    # 如果因为取整导致点数不足，补充随机点
    if len(points) < TOTAL_POINTS:
        points.extend(generate_poisson_process_points(lambda_intensity=0.5, area_x=AREA_X, area_y=AREA_Y, num_points_to_generate=TOTAL_POINTS - len(points)))
    
    # 如果点数过多，则截断
    points = points[:TOTAL_POINTS]
    return format_points_to_json(points, AREA_X, AREA_Y)


# 3. 网格化分布 (农业)
# alpha -> 1. D(x,y) = sum(1/(mn) * delta(x-xi, y-yj))
# 近似 delta 为小方差高斯或直接网格点加微小抖动
def generate_agriculture_scenario():
    alpha = 1.0 # 纯网格
    # 确定 m 和 n，使得 m*n 大约等于 TOTAL_POINTS
    # 假设区域是方形的，我们尝试让 m 和 n 接近 sqrt(TOTAL_POINTS)
    side = int(np.sqrt(TOTAL_POINTS))
    rows = side
    cols = side
    if rows * cols < TOTAL_POINTS: # 如果平方数不够，增加一列或一行
        cols +=1
        if rows * cols < TOTAL_POINTS and rows < cols: # 尝试增加行数
             rows +=1
        elif rows * cols < TOTAL_POINTS and cols < rows: # 尝试增加列数
             cols +=1


    # 确保至少有 TOTAL_POINTS 个点，多余的点会被截断
    while rows * cols < TOTAL_POINTS:
        if rows <= cols:
            rows += 1
        else:
            cols += 1
            
    num_actual_grid_points = rows * cols
    
    spacing_x = AREA_X / (cols +1) # 留出一些边距
    spacing_y = AREA_Y / (rows +1)
    
    # 使用非常小的抖动来模拟狄拉克函数的集中性
    points = generate_grid_points(rows, cols, spacing_x, spacing_y, AREA_X, AREA_Y, jitter=1.0)
    
    points = points[:TOTAL_POINTS] # 确保不多于 TOTAL_POINTS
    return format_points_to_json(points, AREA_X, AREA_Y)

# 4. 结构化线性分布 (工厂)
# alpha = 1. D(x,y) = sum( (c^-2.5 / Z) * N(mu_c, diag(sig_x^2, sig_y^2)) )
def generate_factory_scenario():
    alpha = 1.0
    num_linear_structures = 3 # 假设 L=3 条线性结构
    points_per_structure_ideal = TOTAL_POINTS // num_linear_structures
    
    all_points = []
    assigned_points = 0

    # 计算归一化常数 Z (这里简化，假设权重直接分配点数)
    # 权重 c^-2.5 (c从1开始)
    raw_weights = [(c+1)**(-2.5) for c in range(num_linear_structures)]
    Z = sum(raw_weights)
    normalized_weights = [w / Z for w in raw_weights]

    for i in range(num_linear_structures):
        # 为每条线性结构分配点数
        if i < num_linear_structures - 1:
            num_points_for_this_structure = int(normalized_weights[i] * TOTAL_POINTS)
        else:
            num_points_for_this_structure = TOTAL_POINTS - assigned_points # 最后一个结构补齐

        if num_points_for_this_structure <= 0 and TOTAL_POINTS > assigned_points : # 至少分配一个点如果还有剩余
             num_points_for_this_structure = 1
        elif num_points_for_this_structure <=0:
            continue


        # 定义线性结构的中心线 (可以是水平、垂直或倾斜的)
        # 这里简化为几条水平或垂直的线段
        is_horizontal = random.choice([True, False])
        if is_horizontal:
            line_y = random.uniform(0.1 * AREA_Y, 0.9 * AREA_Y)
            line_start_x = random.uniform(0.1 * AREA_X, 0.3 * AREA_X)
            line_end_x = random.uniform(0.7 * AREA_X, 0.9 * AREA_X)
            
            mu_start = [line_start_x, line_y]
            mu_end = [line_end_x, line_y]
            # 各向异性协方差，沿x轴拉伸
            sigma_x_sq = random.uniform(50, 100)**2 # 沿线方向的较大方差
            sigma_y_sq = random.uniform(5, 15)**2   # 垂直线方向的较小方差
            covariance_matrix = [[sigma_x_sq, 0], [0, sigma_y_sq]]
        else: # 垂直
            line_x = random.uniform(0.1 * AREA_X, 0.9 * AREA_X)
            line_start_y = random.uniform(0.1 * AREA_Y, 0.3 * AREA_Y)
            line_end_y = random.uniform(0.7 * AREA_Y, 0.9 * AREA_Y)

            mu_start = [line_x, line_start_y]
            mu_end = [line_x, line_end_y]
            # 各向异性协方差，沿y轴拉伸
            sigma_x_sq = random.uniform(5, 15)**2   # 垂直线方向的较小方差
            sigma_y_sq = random.uniform(50, 100)**2 # 沿线方向的较大方差
            covariance_matrix = [[sigma_x_sq, 0], [0, sigma_y_sq]]

        # 在这条线段上生成多个高斯集群的中心点 mu_c
        # 为了简化，我们假设每个结构是一个大的、拉长的高斯分布
        # 其中心是线段的中点
        effective_mu = [(mu_start[0] + mu_end[0]) / 2, (mu_start[1] + mu_end[1]) / 2]
        
        # 如果需要模拟多个小集群组成线状，则需要更复杂的 mu_c 生成逻辑
        # 这里简化为一个拉长的高斯分布代表一个线性结构
        structure_points = generate_gaussian_cluster_points(effective_mu, covariance_matrix, num_points_for_this_structure, AREA_X, AREA_Y)
        all_points.extend(structure_points)
        assigned_points += len(structure_points) # 用实际生成的点数更新

    # 如果因为取整导致点数不足或过多
    if len(all_points) < TOTAL_POINTS:
        all_points.extend(generate_poisson_process_points(lambda_intensity=0.1, area_x=AREA_X, area_y=AREA_Y, num_points_to_generate=TOTAL_POINTS - len(all_points)))
    
    all_points = all_points[:TOTAL_POINTS]
    return format_points_to_json(all_points, AREA_X, AREA_Y)


# 5. 自定义混合分布场景 (例如：背景随机点 + 几个不同方向和大小的线性集群 + 少量小型密集集群)
def generate_custom_hybrid_scenario():
    points = []
    
    # 参数
    alpha_random = 0.2  # 20% 随机背景点
    alpha_linear_clusters = 0.5 # 50% 线性集群点
    alpha_small_dense_clusters = 0.3 # 30% 小型密集集群点

    num_random_bg_points = int(alpha_random * TOTAL_POINTS)
    num_linear_points_total = int(alpha_linear_clusters * TOTAL_POINTS)
    num_small_dense_points_total = TOTAL_POINTS - num_random_bg_points - num_linear_points_total

    # a. 生成随机背景点
    points.extend(generate_poisson_process_points(lambda_intensity=0.2, area_x=AREA_X, area_y=AREA_Y, num_points_to_generate=num_random_bg_points))

    # b. 生成线性集群点
    num_linear_structures = 2 # 例如2条线性结构
    if num_linear_structures > 0 and num_linear_points_total > 0:
        points_per_linear_structure = num_linear_points_total // num_linear_structures
        assigned_linear_points = 0
        for i in range(num_linear_structures):
            num_points_for_this = points_per_linear_structure if i < num_linear_structures -1 else num_linear_points_total - assigned_linear_points
            if num_points_for_this <=0: continue

            # 随机方向和长度的线性集群
            angle = random.uniform(0, np.pi) # 0 to 180 degrees
            length = random.uniform(AREA_X * 0.3, AREA_X * 0.7)
            
            # 随机中心点
            center_x = random.uniform(length/2 * 0.8, AREA_X - length/2 * 0.8) # 确保线段在界内
            center_y = random.uniform(length/2 * 0.8, AREA_Y - length/2 * 0.8) # 粗略估计
            
            # 定义各向异性协方差矩阵以匹配方向和长度
            # 主轴方差 (沿长度方向)
            sigma_major_sq = (length / 2)**2 / 9 # 使大部分点落在长度内 (3 sigma rule)
            # 次轴方差 (垂直于长度方向，使其 "细长")
            sigma_minor_sq = random.uniform(5, 15)**2
            
            # 旋转矩阵
            R = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
            # 未旋转的协方差矩阵
            D = np.array([[sigma_major_sq, 0],
                          [0, sigma_minor_sq]])
            # 旋转后的协方差矩阵
            covariance_matrix = R @ D @ R.T
            
            linear_cluster_points = generate_gaussian_cluster_points([center_x, center_y], covariance_matrix, num_points_for_this, AREA_X, AREA_Y)
            points.extend(linear_cluster_points)
            assigned_linear_points += len(linear_cluster_points)


    # c. 生成小型密集集群点
    num_small_clusters = 3 # 例如3个小型密集集群
    if num_small_clusters > 0 and num_small_dense_points_total > 0:
        points_per_small_cluster = num_small_dense_points_total // num_small_clusters
        assigned_small_dense_points = 0
        for i in range(num_small_clusters):
            num_points_for_this = points_per_small_cluster if i < num_small_clusters -1 else num_small_dense_points_total - assigned_small_dense_points
            if num_points_for_this <=0: continue

            mu_c = [random.uniform(0.1 * AREA_X, 0.9 * AREA_X), random.uniform(0.1 * AREA_Y, 0.9 * AREA_Y)]
            sigma_c = random.uniform(5, 15) # "密集"意味着较小的 sigma
            small_dense_points = generate_gaussian_cluster_points(mu_c, sigma_c, num_points_for_this, AREA_X, AREA_Y)
            points.extend(small_dense_points)
            assigned_small_dense_points += len(small_dense_points)

    # d. 补齐或截断点数
    if len(points) < TOTAL_POINTS:
        points.extend(generate_poisson_process_points(lambda_intensity=0.1, area_x=AREA_X, area_y=AREA_Y, num_points_to_generate=TOTAL_POINTS - len(points)))
    
    points = points[:TOTAL_POINTS]
    return format_points_to_json(points, AREA_X, AREA_Y)


if __name__ == "__main__":
    base_path = "showcases/Simulating_a_data_collection_scenario/"

    # 生成并保存每种场景
    forest_data = generate_forest_scenario()
    save_to_json(forest_data, base_path + "sensor_locations_forest.json")

    urban_data = generate_urban_scenario()
    save_to_json(urban_data, base_path + "sensor_locations_urban.json")

    agriculture_data = generate_agriculture_scenario()
    save_to_json(agriculture_data, base_path + "sensor_locations_agriculture.json")

    factory_data = generate_factory_scenario()
    save_to_json(factory_data, base_path + "sensor_locations_factory.json")
    
    custom_hybrid_data = generate_custom_hybrid_scenario()
    save_to_json(custom_hybrid_data, base_path + "sensor_locations_custom_hybrid.json")

    print(f"All sensor location files generated in {base_path}")
    print(f"Total points per file (target): {TOTAL_POINTS}")
    print(f"Forest: {len(forest_data)} points")
    print(f"Urban: {len(urban_data)} points")
    print(f"Agriculture: {len(agriculture_data)} points")
    print(f"Factory: {len(factory_data)} points")
    print(f"Custom Hybrid: {len(custom_hybrid_data)} points")