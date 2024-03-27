import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
from sortedcontainers import SortedSet
import copy
import os
from os import path

class BoardSolver:
    def __init__(self, board: np.ndarray):
        self.board = board
        self.available_steps = SortedSet([])
        self.steps_taken = []
        self.step_num = 0
        self.max_step_num = np.sum(self.board == 1) - 1
        empty_coors = np.where(self.board == 0)
        self.last_cell_coors = np.concatenate((empty_coors[0], empty_coors[1]), axis=0)
        self.possible_steps_for_tile = self.build_possible_steps()
        self.blocked_boards = set()
        desired_board = copy.deepcopy(self.board)
        desired_board[desired_board == 1] = 3
        desired_board[desired_board == 0] = 1
        desired_board[desired_board == 3] = 0
        self.desired_board_state = self.encode_board(desired_board)
        self.tile_available_steps(self.last_cell_coors)

    def build_possible_steps(self):
        possible_dirctions = {}
        directions = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if self.board[i, j] != 2:
                    coors = np.array([i, j])
                    temp_steps = []
                    for direction in directions:
                        if np.all((coors + 2 * direction >= 0) & (coors + 2 * direction < len(self.board))):
                            if self.board[coors[0] + 2 * direction[0], coors[1] + 2 * direction[1]] != 2:
                                temp_steps.append([coors, direction])
                                temp_steps.append([coors + 2 * direction, -direction])
                                temp_steps.append([coors + 2 * direction, -direction])
                        if np.all((coors + direction >= 0) & (coors + direction < len(self.board))
                                  & (coors - direction >= 0) & (coors - direction < len(self.board))):
                            if self.board[coors[0] + direction[0], coors[1] + direction[1]] != 2 \
                                    and self.board[coors[0] - direction[0], coors[1] - direction[1]] != 2:
                                temp_steps.append([coors - direction, direction])

                    possible_dirctions[(i, j)] = temp_steps
        return possible_dirctions

    def tile_available_steps(self, tile: np.ndarray):
        for step in self.possible_steps_for_tile[tuple(tile)]:
            if self.can_perform_step(step):
                self.safe_add(step)
            else:
                self.safe_remove(step)

    def can_perform_step(self, step):
        tile, direction = step[0], step[1]
        new_board = copy.deepcopy(self.board)
        new_board[step[0][0], step[0][1]] = 0
        new_board[(step[0] + step[1])[0], (step[0] + step[1])[1]] = 0
        new_board[(step[0] + 2 * step[1])[0], (step[0] + 2 * step[1])[1]] = 1
        if not self.board[tile[0], tile[1]] or not self.board[(tile + direction)[0], (tile + direction)[1]]:
            return False
        if self.board[(tile + 2 * direction)[0], (tile + 2 * direction)[1]]:
            return False
        if self.encode_board(new_board) in self.blocked_boards:
            return False
        return True

    def perform_step(self, step: List[np.ndarray]):
        self.board[step[0][0], step[0][1]] = 0
        self.board[(step[0] + step[1])[0], (step[0] + step[1])[1]] = 0
        self.board[(step[0] + 2 * step[1])[0], (step[0] + 2 * step[1])[1]] = 1
        self.tile_available_steps(step[0])
        self.tile_available_steps(step[0] + step[1])
        self.tile_available_steps(step[0] + 2 * step[1])
        self.step_num += 1

    def retrace_step(self, step: List[np.ndarray]):
        self.board[step[0][0], step[0][1]] = 1
        self.board[(step[0] + step[1])[0], (step[0] + step[1])[1]] = 1
        self.board[(step[0] + 2 * step[1])[0], (step[0] + 2 * step[1])[1]] = 0
        self.tile_available_steps(step[0])
        self.tile_available_steps(step[0] + step[1])
        self.tile_available_steps(step[0] + 2 * step[1])
        self.safe_remove(step)
        self.step_num -= 1

    def safe_remove(self, step):
        encoded = self.encode_step(step)
        if encoded in self.available_steps:
            self.available_steps.remove(encoded)

    def safe_add(self, step):
        encoded = self.encode_step(step)
        if encoded not in self.available_steps:
            self.available_steps.add(encoded)

    def solve_board(self):
        while True:
            step = self.decode_step(self.available_steps.pop(-1))
            self.steps_taken.append(step)
            self.perform_step(step)
            # display_board(step)
            while not self.available_steps:
                self.blocked_boards.add(self.encode_board(self.board))
                if not self.steps_taken:
                    return []
                if self.step_num == self.max_step_num and self.encode_board(self.board) == self.desired_board_state:
                    return self.steps_taken
                last_step = self.steps_taken.pop(-1)
                self.retrace_step(last_step)

    @staticmethod
    def encode_step(step: List[np.ndarray]) -> tuple:
        return tuple(step[0]), tuple(step[1])

    @staticmethod
    def decode_step(key: tuple) -> List[np.ndarray]:
        return [np.array(key[0]), np.array(key[1])]

    def encode_board(self, board):
        encoded = str(np.ravel(board))
        encoded = encoded.replace(' ', '').replace('2', '').replace('\n', '')[1:-1]
        return int(encoded, 2)


def display_solution(board: np.array, solution: List[np.array]):
    for step in solution:
        board[step[0][0], step[0][1]] = 0
        board[(step[0] + step[1])[0], (step[0] + step[1])[1]] = 0
        board[(step[0] + 2 * step[1])[0], (step[0] + 2 * step[1])[1]] = 1
        display_board(board, step)


def display_board(board, step: Optional[List]):
    circle_width = 0.2
    fig, ax = plt.subplots()
    ax.add_patch(plt.Circle((len(board[0]) // 2, -len(board) // 2 + 1), len(board) / 2 + 0.5, linewidth=2, edgecolor='k', facecolor='tab:brown', fill=True))
    ax.add_patch(plt.Circle((len(board[0]) // 2, -len(board) // 2 + 1), len(board) / 2 + 0.1, linewidth=2, edgecolor='k', facecolor='tab:brown', fill=True))
    for i, row in enumerate(board):
        for j, tile in enumerate(row):
            if tile == 1:
                ax.add_patch(plt.Circle((j, -i), circle_width, edgecolor='k', facecolor='tab:blue'))
    if step is not None:
        ax.add_patch(plt.Circle((step[0][1], -step[0][0]), circle_width, color='k', fill=False))
        ax.add_patch(plt.Circle((step[0][1] + step[1][1], -step[0][0] - step[1][0]), circle_width, color='k', fill=False))
        ax.arrow(step[0][1], -step[0][0], 2 * step[1][1], -2 * step[1][0], head_width=0.2, head_length=0.2, color='k')
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.autoscale()
    ax.set_aspect('equal')
    plt.show()


if __name__ == '__main__':
    load_solution = True
    file_name = f'solution.npy'
    board = np.array([[2, 2, 1, 1, 1, 2, 2],
                      [2, 2, 1, 1, 1, 2, 2],
                      [1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 0, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1],
                      [2, 2, 1, 1, 1, 2, 2],
                      [2, 2, 1, 1, 1, 2, 2]], np.int32)
    if load_solution:
        solution = np.load(path.join(os.getcwd(), file_name))
        display_solution(board, solution)
    else:
        solver = BoardSolver(board)
        solution = solver.solve_board()
        np.save(path.join(os.getcwd(), file_name), solution)
    print(solution)