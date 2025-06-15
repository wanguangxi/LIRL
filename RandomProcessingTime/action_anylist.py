import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm

import matplotlib.pyplot as plt

# 替换为你的.npy文件路径
npy_file = 'job5_action_records_0.2_0.8_co_2000.npy'

# 读取.npy文件
data = np.load(npy_file)[0:2000]

# # 打印数据的形状和类型，便于调试
# print("数据形状:", data.shape)
# print("数据类型:", data.dtype)

# # 如果数据为 (m, n, 3)，绘制每个球内n种数据类型随迭代次数的变化3D视图
# if data.ndim == 3 and data.shape[2] == 3:
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection='3d')
#     m, n, _ = data.shape
#     X = np.arange(m)  # 迭代次数
#     Y = np.arange(n)  # 数据类型
#     X, Y = np.meshgrid(X, Y)
#     # 绘制每个数据类型的球内点轨迹
#     for j in range(n):
#         ax.plot(
#             data[:, j, 0],  # x
#             data[:, j, 1],  # y
#             data[:, j, 2],  # z
#             label=f'Type {j}'
#         )
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('3D Trajectory of Each Data Type in a Sphere')
#     ax.legend()
# else:
#     print("数据不是 (m, n, 3) 形状，无法绘制。")



# 绘制 n 个图形，表示每个决策类型的 3 个决策结果在（0,1）之间的正态分布曲线

# if data.ndim == 3 and data.shape[2] == 3:
#     n = data.shape[1]
#     fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n))
#     if n == 1:
#         axes = [axes]
#     x = np.linspace(0, 1, 200)
#     for j in range(n):
#         for k in range(3):
#             vals = data[:, j, k]
#             mu, std = np.mean(vals), np.std(vals)
#             pdf = norm.pdf(x, mu, std)
#             axes[j].plot(x, pdf, label=f'Result {k} (μ={mu:.2f}, σ={std:.2f})')
#         axes[j].set_ylabel(f'Type {j}')
#         axes[j].legend()
#     axes[-1].set_xlabel('Value')
#     plt.suptitle('Normal Distribution of 3 Decision Results for Each Type')
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
# else:
#     print("数据不是 (m, n, 3) 形状，无法绘制正态分布曲线。")



# 绘制 n 个图形，表示每个决策类型的 3 个决策结果随训练次数的变化
if data.ndim == 3 and data.shape[2] == 3:
    n = data.shape[1]
    m = data.shape[0]
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    x = np.arange(m)
    for j in range(n):
        for k in range(3):
            axes[j].plot(x, data[:, j, k], label=f'Result {k}')
        axes[j].set_ylabel(f'Type {j}')
        axes[j].legend()
    axes[-1].set_xlabel('Training Step')
    plt.suptitle('Decision Results over Training Steps for Each Type')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
else:
    print("数据不是 (m, n, 3) 形状，无法绘制决策结果随训练次数的变化。")
    
# # 绘制 m=100,200,500,1000 时的切片（假设 m=data.shape[0]，n=data.shape[1]，每个切片为 data[m_idx, :, :]）
# slice_indices = [100, 200, 500, 999]
# if data.ndim == 3 and data.shape[2] == 3:
#     fig = plt.figure(figsize=(20, 8))
#     for idx, m_idx in enumerate(slice_indices):
#         if m_idx < data.shape[0]:
#             ax = fig.add_subplot(1, 4, idx+1, projection='3d')
#             X = np.arange(data.shape[1])
#             Y = np.arange(3)  # 3 channels
#             X, Y = np.meshgrid(X, Y)
#             Z = data[m_idx, :, :].T  # shape (3, n)
#             surf = ax.plot_surface(X, Y, Z, cmap='viridis')
#             ax.set_xlabel('Decision step')
#             ax.set_ylabel('Latent variable for task selection')
#             ax.set_zlabel('Latent variable for machine selection')
#             ax.set_title(f'Slice m={m_idx}')
#             fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
#         else:
#             print(f"索引 {m_idx} 超出数据范围，跳过。")
#     plt.suptitle('Slices at m=100,200,500,1000')
# else:
#     print("数据不是 (m, n, 3) 形状，无法绘制切片。")


# # 打印数据的形状和类型
# print("数据形状:", data.shape)

# # 打印数据
# print(data.ndim)

# if data.ndim == 2:
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     X = np.arange(data.shape[1])
#     Y = np.arange(data.shape[0])
#     X, Y = np.meshgrid(X, Y)
#     Z = data
#     ax.plot_surface(X, Y, Z, cmap='viridis')
#     ax.set_xlabel('Column')
#     ax.set_ylabel('Row')
#     ax.set_zlabel('Value')
#     plt.title('3D Surface Plot of Data')
# else:
#     print("数据不是二维，无法绘制3D立体图。")

plt.show()

# # 判断数据维度并绘图
# if data.ndim == 1:
#     plt.plot(data)
#     plt.xlabel('Index')
#     plt.ylabel('Value')
# elif data.ndim == 2:
#     plt.imshow(data, aspect='auto', cmap='viridis')
#     plt.colorbar()
#     plt.xlabel('Column')
#     plt.ylabel('Row')
# else:
#     print("数据维度大于2，无法直接绘制。")
#     exit()

# plt.title('Data from {}'.format(npy_file))
# plt.show()