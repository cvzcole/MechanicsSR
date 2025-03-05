import itertools
import numpy as np
import os

def matrix_column_combinations(matrix, x, output_dir):
    # 获取矩阵的列数
    n_cols = matrix.shape[1]-1
    
    # 检查x是否大于列数
    if x > n_cols:
        raise ValueError("x cannot be greater than the number of columns in the matrix")

    # 生成所有列组合的索引
    column_indices = range(n_cols)
    column_combinations = itertools.combinations(column_indices, x)

    # 创建目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    combination_matrices = []

    # 保存每个组合到不同的txt文件
    for index, comb in enumerate(column_combinations):
        #comb = (np.column_stack((comb, matrix[:, -1])))
        combination_matrix = np.column_stack((matrix[:, list(comb)], matrix[:, -1]))
        combination_matrices.append(combination_matrix)

        # 生成文件名并保存
        filename = os.path.join(output_dir, f'combination_{index}.txt')
        np.savetxt(filename, combination_matrix, delimiter=' ', fmt='%f')

    return combination_matrices

# 示例
n, m = 4, 4  # 矩阵维度
matrix = np.random.randint(1, 10, (n, m))/2.3  # 创建一个 4x3 的随机矩阵
x = 2  # 选择的列数
output_directory = "combinations_output"  # 指定保存文件的目录

# 获取所有基于 x 的列组合，并保存到txt文件
combinations = matrix_column_combinations(matrix, x, output_directory)

# 打印结果
print("Original Matrix:\n", matrix)
print("\nColumn Combinations have been saved in the directory:", output_directory)

# 011524 update