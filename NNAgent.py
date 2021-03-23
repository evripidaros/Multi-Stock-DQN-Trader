from collections import deque
import random
import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQNNAgent(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.lr = 0.1
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01

        # Model
        self.model = self.build_model1()

    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        print(model.summary())
        return model

    def build_model1(self):
        X = keras.layers.Input(shape=(self.state_size,))
        _fc = keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform')(X)
        _fc = keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform')(_fc)
        y = keras.layers.Dense(self.action_size, activation='linear', kernel_initializer='he_uniform')(_fc)
        model = keras.models.Model(X, y)
        # model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.lr))
        model.compile(loss='mse', optimizer=Adam())
        model.summary()
        return model

    # def build_model2(self):
    #     model = Sequential()
    #     model.add(Conv1D(filters=1, kernel_size=1, activation='relu', input_shape=(1, 7)))
    #     model.add(MaxPooling1D(pool_size=1))
    #     model.add(Flatten())
    #     model.add(Dense(50, activation='relu'))
    #     model.add(Dense(27))
    #     model.compile(optimizer='adam', loss='mse')
    #     return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            # print(self.epsilon)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # print('explore====random action')
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        # print(str(act_values))
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size=32):
        """ vectorized implementation; 30x speed up compared with for loop """
        mini_batch = random.sample(self.memory, batch_size)

        states = np.array([tup[0][0] for tup in mini_batch])
        actions = np.array([tup[1] for tup in mini_batch])
        rewards = np.array([tup[2] for tup in mini_batch])
        next_states = np.array([tup[3][0] for tup in mini_batch])
        done = np.array([tup[4] for tup in mini_batch])

        # Q(s', a)
        target = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1)
        # end state target is reward itself (no lookahead)
        target[done] = rewards[done]

        # Q(s, a)
        target_f = self.model.predict(states)
        # make the agent to approximately map the current state to future discounted reward
        target_f[range(batch_size), actions] = target

        self.model.fit(states, target_f, epochs=1, verbose=0)

        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        #     print(self.epsilon)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
