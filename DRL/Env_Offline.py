import numpy as np
import pandas as pd


class Env():
    def __init__(self):
        self.all_round_state = []
        self.completed_round = 0
        self.completed_epoch = 0
        self.ROUND_NUM = 12
        self.EPOCH_NUM = 20
        self.init_data()

    def init_round(self):
        self.completed_round = 0
        self.completed_epoch = 0
        return self.get_state()

    def init_epoch(self):
        self.completed_epoch = 0
        return self.get_state()

    def step(self, action):
        is_stop = False
        if action == 1 or self.completed_epoch == self.EPOCH_NUM - 1:
            is_stop = True
        '''reward=（这回合精度-上回合精度）-这回合花费的时间，1和0.0001用于加权'''
        if self.completed_epoch == 0:
            # reward = (1 * (self.all_round_state[self.completed_round][self.completed_epoch][5]) -
            #           0.1 * self.all_round_state[self.completed_round][self.completed_epoch][1])
            reward = self.all_round_state[self.completed_round][self.completed_epoch][4] - 0.01
        else:
            # print(self.all_round_state[self.completed_round][self.completed_epoch][1])
            # print("acc increace={}".format(self.all_round_state[self.completed_round][self.completed_epoch][5] -
            #                self.all_round_state[self.completed_round][self.completed_epoch - 1][5]))
            reward = self.all_round_state[self.completed_round][self.completed_epoch][4] - \
                     self.all_round_state[self.completed_round][self.completed_epoch - 1][4] - 0.01
        if self.completed_epoch != self.EPOCH_NUM - 1:
            self.completed_epoch += 1

        # print("action={},reward={}".format(is_stop,round(reward,3)))
        return self.get_state(), reward, is_stop

    def get_state(self):
        state = np.array(self.all_round_state[self.completed_round][self.completed_epoch])
        # print(f"[TEST] state={state}")
        return state

    def get_expert_action(self, state):
        expert_action = 0
        if self.completed_epoch == 0:
            tmp_reward = self.all_round_state[self.completed_round][self.completed_epoch][5]
        else:
            tmp_reward = self.all_round_state[self.completed_round][self.completed_epoch][5] - \
                         self.all_round_state[self.completed_round][self.completed_epoch - 1][5]
        '''读取当前epoch的精度提升，如果小于时间成本就停止执行，expert_action=1表示stop'''
        if tmp_reward < 0.1 * self.all_round_state[self.completed_round][self.completed_epoch][1]:
            expert_action = 1
        return expert_action

    def normalize_array(self, data_array, min_vals, max_vals):
        """对输入数组按列进行Min-Max归一化"""
        return (data_array - min_vals) / (max_vals - min_vals + 1e-8)  # 加1e-8防除0

    def init_data(self):
        dataset_path = r'DRL_data.xlsx'
        raw_data = pd.read_excel(dataset_path, header=0, sheet_name="Sheet1")
        # features = ["epoch", "time", "box_loss", "cls_loss", "dfl_loss", "mAP50", "mAP90"]
        features = ["epoch", "box_loss", "cls_loss", "dfl_loss", "mAP50", "mAP90"]
        # print(raw_data.columns.tolist())
        all_data = raw_data[features].values
        min_vals = all_data.min(axis=0)
        max_vals = all_data.max(axis=0)
        # print(min_vals)
        # print(max_vals)
        for i in range(self.ROUND_NUM):
            tmp_epoch_state = []
            for j in range(self.EPOCH_NUM):
                row = np.array([
                    raw_data.loc[i * self.EPOCH_NUM + j]["epoch"],
                    # raw_data.loc[i * self.EPOCH_NUM + j]["time"],
                    raw_data.loc[i * self.EPOCH_NUM + j]["box_loss"],
                    raw_data.loc[i * self.EPOCH_NUM + j]["cls_loss"],
                    raw_data.loc[i * self.EPOCH_NUM + j]["dfl_loss"],
                    raw_data.loc[i * self.EPOCH_NUM + j]["mAP50"],
                    raw_data.loc[i * self.EPOCH_NUM + j]["mAP90"]
                ])
                norm_row = self.normalize_array(row, min_vals, max_vals)
                tmp_epoch_state.append(norm_row)
            self.all_round_state.append(tmp_epoch_state)
