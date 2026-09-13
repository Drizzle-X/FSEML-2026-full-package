"""Dependency-free protocol helpers shared with reproduction-package tests."""
import math
import random


def class_order(num_classes, seed):
    order = list(range(num_classes))
    random.Random(seed).shuffle(order)
    return order


def metrics_from_matrix(matrix, learned_tasks=None):
    k = learned_tasks or len(matrix)
    final_row = matrix[k - 1][:k]
    acc = sum(final_row) / k
    la = sum(matrix[index][index] for index in range(k)) / k
    if k == 1:
        fm = 0.0
    else:
        forgetting = []
        for task_index in range(k - 1):
            previous = [matrix[row][task_index] for row in range(task_index, k - 1)]
            forgetting.append(max(previous) - matrix[k - 1][task_index])
        fm = sum(forgetting) / len(forgetting)
    return {"ACC": float(acc), "FM": float(fm), "LA": float(la)}
