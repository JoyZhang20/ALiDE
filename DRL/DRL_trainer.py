import collections
import heapq
from DRL.PPO import PPO
from Env_Offline import Env
from tqdm import tqdm
import matplotlib.pyplot as plt
from tool.JoyTool import write_list_to_txt
PPO_kwargs = {
    "state_dim": 6,
    "action_dim": 2,
    "ppo_hidden_num": 128,
    "lr_actor": 0.0001,
    "lr_critic": 0.0001,
    "gamma": 0.99,
    "K_epochs": 80,
    "eps_clip": 0.2,
    "has_continuous_action_space": False,
    "action_std_init": 0.6,
}

def show_pyplot(record_y, label="reward"):
    plt.figure()
    plt.plot(record_y, color='r', label=label, linewidth=2.5, linestyle='-')  # 红虚线为损失值
    plt.xlabel("Step")
    plt.ylabel(label)
    plt.legend()
    plt.show()


def trainPPO():
    env = Env()
    ppo_agent = PPO(**PPO_kwargs)
    # ppo_agent.load("ppo_base.pth")
    all_episode_reward = []
    DRL_EPISODE = 300
    pbar = tqdm(range(DRL_EPISODE), ncols=120)
    for episode in pbar:
        env.init_round()
        epoch_reward = 0
        for r in range(env.ROUND_NUM):
            state = env.init_epoch()
            for e in range(env.EPOCH_NUM):
                action = ppo_agent.select_action(state)
                '''在前一半的round中使用离线搜索实现模仿学习[这个组件效果非常好，可以通过关闭这几行代码验证一下]'''
                if episode<0.9*DRL_EPISODE:
                # if r < 0.5 * env.ROUND_NUM:
                    expert_action = env.get_expert_action(state)
                    ppo_agent.expert_data.add_expert_data(state, expert_action)
                next_state, reward, is_stop = env.step(action)
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(not is_stop)
                state = next_state
                epoch_reward += reward
                if is_stop:
                    break
        epoch_reward = round(epoch_reward, 2)
        all_episode_reward.append(epoch_reward)
        if len(ppo_agent.buffer.states) > 10:
            ppo_agent.update()
        pbar.set_postfix(episode=episode, reward=str(epoch_reward), best_reawrd=str(round(max(all_episode_reward), 5)))
    show_pyplot(all_episode_reward)
    ppo_agent.save("ppo.pth")
    write_list_to_txt(all_episode_reward)

def evalPPO():
    env = Env()
    ppo_agent = PPO(**PPO_kwargs)
    ppo_agent.load("ppo.pth")
    env.init_round()
    epoch_reward = 0
    for r in range(env.ROUND_NUM):
        state = env.init_epoch()
        for e in range(env.EPOCH_NUM):
            action = ppo_agent.select_action(state)
            next_state, reward, is_stop = env.step(action)
            state = next_state
            epoch_reward += reward
            if is_stop:
                print(f"round={r},epoch={e}")
                break
    epoch_reward = round(epoch_reward, 2)

if __name__ == '__main__':
    trainPPO()
    # evalPPO()
