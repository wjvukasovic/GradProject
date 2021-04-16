# Importing needed packages for learning
import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import io
import base64
import time
from tensorflow.contrib.layers import flatten, conv2d, fully_connected
from collections import deque, Counter
from gym import logger as gymlogger
from gym.wrappers import Monitor
from IPython.display import HTML
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display

gymlogger.set_level(40)

tf.compat.v1.disable_eager_execution()

print(tf.__version__)

color = np.array([210, 164, 74]).mean()

# parameters for epsilon greedy function
epsilon = 0.5
eps_min = 0.05
eps_max = 1.0
eps_decay_steps = 500000

# initializing buffer

buffer_len = 20000

exp_buffer = deque(maxlen=buffer_len)

# parameters for our training process
num_episodes = 400
batch_size = 48
input_shape = (None, 88, 80, 1)

learning_rate = 0.001
X_shape = (None, 88, 80, 1)
discount_factor = 0.97

global_step = 0
copy_steps = 100
steps_train = 4
start_steps = 2000

# preprocessing function
# used to crop the image from the environment and convert them into
# one-dimensional tensors
def preprocess_observation(obs):

    img = obs[1:176:2, ::2]

    img = img.mean(axis=2)

    img[img==color] = 0

    img = (img - 128) / 128 - 1

    return img.reshape(88,80,1)

# defining q network
# 3 layer convolutional network
# takes in preprocessed images and flattens them
# before feeding them to a fully connected layer
# It then outputs the probabilities of taking each action
# in the game space
def q_network(X, name_scope):

    initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=2.0)

    with tf.compat.v1.variable_scope(name_scope) as scope:

        layer_1 = conv2d(X, num_outputs=32, kernel_size=(8, 8), stride=4, padding='SAME',
                         weights_initializer=initializer)
        tf.compat.v1.summary.histogram('layer_1', layer_1)

        layer_2 = conv2d(layer_1, num_outputs=64, kernel_size=(4, 4), stride=2, padding='SAME',
                         weights_initializer=initializer)
        tf.compat.v1.summary.histogram('layer_2', layer_2)

        layer_3 = conv2d(layer_2, num_outputs=64, kernel_size=(3, 3), stride=1, padding='SAME',
                         weights_initializer=initializer)
        tf.compat.v1.summary.histogram('layer_3', layer_3)

        flat = flatten(layer_3)

        fc = fully_connected(flat, num_outputs=128, weights_initializer=initializer)
        tf.compat.v1.summary.histogram('fc', fc)

        output = fully_connected(fc, num_outputs=n_outputs, activation_fn=None, weights_initializer=initializer)
        tf.compat.v1.summary.histogram('output', output)

        vars = {v.name[len(scope.name):]: v for v in
                tf.compat.v1.get_collection(key=tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)}

        return vars, output

# epsilon greedy function to ensure we visit every possible state-action combination
def epsilon_greedy(action, step):
    p = np.random.random(1).squeeze()
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps) #Decaying policy with more steps
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        return action

def sample_memories(batch_size):
    perm_batch = np.random.permutation(len(exp_buffer))[:batch_size]
    mem = np.array(exp_buffer)[perm_batch]
    return mem[:,0], mem[:,1], mem[:,2], mem[:,3], mem[:,4]

def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

def wrap_env(env):
    env = Monitor(env, './video', force=True)
    return env


# initializing the gym environment
# will print a few screens of gameplay
# and display the actions available to the agent
env = gym.make("KungFuMaster-v0")
n_outputs = env.action_space.n
print(n_outputs)
print(env.env.get_action_meanings())

observation = env.reset()

for i in range(22):

    if i > 20:
        plt.imshow(observation)
        plt.show()

    observation, _, _, _ = env.step(1)



obs_preprocessed = preprocess_observation(observation).reshape(88,80)
plt.imshow(obs_preprocessed)
plt.show()
print(observation.shape)
print(obs_preprocessed.shape)

tf.compat.v1.reset_default_graph()



logdir = 'logs'
tf.compat.v1.reset_default_graph()

X = tf.compat.v1.placeholder(tf.float32, shape=X_shape)

in_training_mode = tf.compat.v1.placeholder(tf.bool)



mainQ, mainQ_outputs = q_network(X, 'mainQ')

targetQ, targetQ_outputs = q_network(X, 'targetQ')


X_action = tf.compat.v1.placeholder(tf.int32, shape=(None,))
Q_action = tf.reduce_sum(input_tensor=targetQ_outputs * tf.one_hot(X_action, n_outputs), axis=-1, keepdims=True)

copy_op = [tf.compat.v1.assign(main_name, targetQ[var_name]) for var_name, main_name in mainQ.items()]
copy_target_to_main = tf.group(*copy_op)



# defining loss function
# squared difference of our target action and our predicted action

y = tf.compat.v1.placeholder(tf.float32, shape=(None,1))

loss = tf.reduce_mean(input_tensor=tf.square(y - Q_action))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.compat.v1.global_variables_initializer()

loss_summary = tf.compat.v1.summary.scalar('LOSS', loss)
merge_summary = tf.compat.v1.summary.merge_all()
file_writer = tf.compat.v1.summary.FileWriter(logdir, tf.compat.v1.get_default_graph())

training_start = time.time()
with tf.compat.v1.Session() as sess:
    init.run()

    history = []
    for i in range(num_episodes):
        done = False
        obs = env.reset()
        epoch = 0
        episodic_reward = 0
        actions_counter = Counter()
        episodic_loss = []

        while not done:

            obs = preprocess_observation(obs)


            actions = mainQ_outputs.eval(feed_dict={X: [obs], in_training_mode: False})

            action = np.argmax(actions, axis=-1)
            actions_counter[str(action)] += 1

            action = epsilon_greedy(action, global_step)

            next_obs, reward, done, _ = env.step(action)

            exp_buffer.append([obs, action, preprocess_observation(next_obs), reward, done])

            if global_step % steps_train == 0 and global_step > start_steps:

                o_obs, o_act, o_next_obs, o_rew, o_done = sample_memories(batch_size)

                o_obs = [x for x in o_obs]

                o_next_obs = [x for x in o_next_obs]

                next_act = mainQ_outputs.eval(feed_dict={X: o_next_obs, in_training_mode: False})

                y_batch = o_rew + discount_factor * np.max(next_act, axis=-1) * (1 - o_done)

                mrg_summary = merge_summary.eval(
                    feed_dict={X: o_obs, y: np.expand_dims(y_batch, axis=-1), X_action: o_act, in_training_mode: False})
                file_writer.add_summary(mrg_summary, global_step)

                train_loss, _ = sess.run([loss, training_op],
                                         feed_dict={X: o_obs, y: np.expand_dims(y_batch, axis=-1), X_action: o_act,
                                                    in_training_mode: True})
                episodic_loss.append(train_loss)

            if (global_step + 1) % copy_steps == 0 and global_step > start_steps:
                copy_target_to_main.run()

            obs = next_obs
            epoch += 1
            global_step += 1
            episodic_reward += reward

        history.append(episodic_reward)
        print('Epochs per episode:', epoch, 'Episode Reward:', episodic_reward, "Episode number:", len(history))

    plt.plot(history)
    plt.show()

training_stop = time.time()
print(training_stop - training_start)

display = Display(visible=0, size=(1400, 900))
display.start()


env = wrap_env(gym.make('KungFuMaster-v0'))
observation = env.reset()
new_observation = observation

prev_input = None
done = False

with tf.compat.v1.Session() as sess:
    init.run()
    while True:
        if True:

            obs = preprocess_observation(observation)

            actions = mainQ_outputs.eval(feed_dict={X: [obs], in_training_mode: False})

            action = np.argmax(actions, axis=-1)
            actions_counter[str(action)] += 1

            action = epsilon_greedy(action, global_step)
            env.render()
            observation = new_observation
            new_observation, reward, done, _ = env.step(action)

            if done:
                break

    env.close()
    show_video()