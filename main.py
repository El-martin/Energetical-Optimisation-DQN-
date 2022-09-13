""" Main Loop"""
import time
import os
import pickle
from DQN import *

epochs = 500
max_timesteps = 1000
data_size = 2000

# setup the training data

os.chdir('C:/Users/Moi/Desktop/Eliot/Travaux/Prépa/Spé/Réseaux_de_neurones_TIPE')
with open('donnees_NW_chaud', 'rb') as fichier:
    mon_unp = pickle.Unpickler(fichier)
    data = mon_unp.load()

i = 30200
data = data[i:i+data_size]
print(f"Starting with : {data[0]}")

# setup the AI & environment

u = HVACunit(70, 15, 1)
env = Environment(15, 20, 25, data, u, [5, 40], [5, 40], [.5, .5, .9, .1])
model = RLAgent(env, 9, 1)


out_file = "Trained_neural_net"

get_saved = input("Load network :")
if get_saved != '':
    with open(get_saved, 'rb') as fichier:
        mon_unp = pickle.Unpickler(fichier)
        loaded_network = mon_unp.load()

    overwrite = input(f"Overwrite {get_saved} in output ? (y/n) ").lower()
    if overwrite == 'y':
        out_file = get_saved
    else:
        out_file = input("Name ouf the new output file :\n>")

    model.layers = loaded_network

# main loop

start = time.time()
fitness_list = []

for i_episode in range(epochs):
    obs = env.reset(15)
    for t in range(max_timesteps):
        # for every new episode
        action = model.select_action(obs)
        prev_obs = obs
        reward, obs, done = env.step(action)

        if i_episode % 10 == 0:
            print(f"action : {action}, state : {obs}, rew : {reward}")

        # Store the agent's experiences
        model.remember(done, action, obs, reward, prev_obs)
        model.experience_replay(30, show_delta=(i_episode % 100 == 0))

        fitness_list.append(reward)
        # if agent lost
        if done:
            print(f"End : action : {action}, state : {obs}")
            print(f"Episode {i_episode} ended after {t+1} time steps\n")
            print(f"epsilon {model.epsilon}")
            break

    # saving the trained network each time
    with open(out_file, 'wb') as fichier:
        pickle.dump(model.layers, fichier)

    with open('fitness_data_5x50', 'wb') as fit_data:
        pickle.dump(fitness_list, fit_data)


duration = ((time.time() - start) // 60, (time.time() - start) % 60)
print(f"Ran {epochs} epochs in {duration[0]} min {duration[1]} sec.")

print(model.select_action(data[0]))
