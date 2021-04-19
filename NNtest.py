import pickle
import argparse
import time
import re
from tqdm import tqdm

from NNEnvironment import MultiEnv, get_data, get_data1, get_scaler, maybe_make_dir
from NNAgent import DQNNAgent

if __name__ == '__main__':
    # config
    num_episodes = 2000
    batch_size = 32
    initial_investment = 20000

    parser = argparse.ArgumentParser()
    # parser.add_argument('-e', '--episode', type=int, default=2000,
    #                    help='number of episode to run')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='batch size for experience replay')
    # parser.add_argument('-i', '--initial_invest', type=int, default=20000,
    #                    help='initial investment amount')
    parser.add_argument('-m', '--mode', type=str, required=False, default="train",
                        help='either "train" or "test"')
    parser.add_argument('-w', '--weights', type=str, help='a trained model weights')
    args = parser.parse_args()

    maybe_make_dir('weights')
    maybe_make_dir('portfolio_val')
    # store a single episode:
    maybe_make_dir('episode_pnl')

    timestamp = time.strftime('%Y%m%d%H%M')

    # data = np.around(get_data())
    # train_data = data[:, :3526]
    # test_data = data[:, 3526:]

    # testing on standard data
    # data = get_data()
    #
    # n_timesteps, n_stocks = data.shape
    #
    # n_train = n_timesteps // 2
    #
    # train_data = data[:n_train]
    # test_data = data[n_train:]

    # testing on a new dataset
    data1 = get_data1()
    n_timesteps, n_stocks = data1.shape
    train_data = data1[:755]
    test_data = data1[755:]

    env = MultiEnv(train_data, initial_investment)

    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNNAgent(state_size, action_size)
    scaler = get_scaler(env)

    portfolio_value = []

    episode_pnl = []

    if args.mode == 'test':
        # remake the env with test data
        env = MultiEnv(test_data, initial_investment)
        # load trained weights
        agent.load(args.weights)
        # when test, the timestamp is same as time when weights was trained
        timestamp = re.findall(r'\d{12}', args.weights)[0]
        num_episodes = 2000
        # make sure epsilon is not 1
        # no need to run multiple episodes if epsilon is 0, it's deterministic
        agent.epsilon = 0.01

    for e in tqdm(range(num_episodes)):
        state = env._reset()
        state = scaler.transform([state])
        for time in range(env.n_step):
            action = agent.act(state)
            next_state, reward, done, info = env._step(action)
            next_state = scaler.transform([next_state])
            # if e + 1 == 1 or e+1 == 10 or e+1 == 100 or (e+1) % 200 == 0:
            if e + 1 == 1:
                episode_pnl.append(float(info['cur_val']))
                if time == env.n_step - 2:
                    with open('episode_pnl/{}-{}-{}.p'.format(timestamp, args.mode, e + 1), 'wb') as fp:
                        pickle.dump(episode_pnl, fp)
                    episode_pnl = []
            # for ticker in tickers:
            #     if episode_pnl.empty:
            #         episode_pnl.append(float(info['cur_val']))
            #     else:
            #         episode_pnl = episode_pnl.join(data[ticker])
            if args.mode == 'train':
                agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, episode end value: {}".format(
                    e + 1, num_episodes, float(info['cur_val'])))
                portfolio_value.append(float(info['cur_val']))  # append episode end portfolio value
                break
            if args.mode == 'train' and len(agent.memory) > args.batch_size:
                agent.replay(args.batch_size)
        if args.mode == 'train' and (e + 1) % 10 == 0:  # checkpoint weights
            agent.save('weights/{}-dqn.h5'.format(timestamp))

    # save portfolio value history to disk
    with open('portfolio_val/{}-{}.p'.format(timestamp, args.mode), 'wb') as fp:
        pickle.dump(portfolio_value, fp)
