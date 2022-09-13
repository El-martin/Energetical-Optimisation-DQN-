""" Getting all the states and actions of one episode """
from DQN import *
from data_processing import *
import matplotlib.pyplot as plt


saved_net = 'best(5x50)_new_env'
max_timesteps = 100

os.chdir('C:/Users/Moi/Desktop/Eliot/Travaux/Prépa/Spé/Réseaux_de_neurones_TIPE')
with open('donnees_NW_chaud', 'rb') as fichier:
    mon_unp = pickle.Unpickler(fichier)
    data = mon_unp.load()

data = data[30200:]
print(data[0:10])

with open(saved_net, 'rb') as fichier:
    mon_unp = pickle.Unpickler(fichier)
    loaded_network = mon_unp.load()


u = HVACunit(70, 15, 1)
env = Environment(20, 20, 25, data, u, [0, 100], [0, 100], [.5, .5, 1, 0.])
model = RLAgent(env, 9, 1)

done = False
obs = env.reset(20)

T_in_list, T_out_list, h_in_list, h_out_list = [], [], [], []
action_list, fitness_list, th_flow_list = [], [], []


for t in range(max_timesteps):
    T_in_list.append(obs[0])
    T_out_list.append(obs[1])
    h_in_list.append(obs[2])
    h_out_list.append(obs[3])

    action = [0, 0, 0, 0, 1]  # model.select_action(obs, best=True)
    reward, obs, done = env.step(action)

    action_list.append(action)
    fitness_list.append(reward)

    th_flow = (obs[1] - obs[0])/1.3e3
    if t % 100 == 0:
        print(f"{t / max_timesteps * 100} completed")


print("Done")

# Graph1
smooth_T_in = smooth_curve(T_in_list, 1)
smooth_T_out = smooth_curve(T_out_list, 1)

fig, ax = plt.subplots()
ax.set_xlabel('Time')
ax.set_ylabel('Temperature')
plt.plot(smooth_T_in, label='T int °C')
plt.plot(smooth_T_out, label='T ext °C')

plt.legend()
plt.show()

# Graph2
ax.set_xlabel('itération')

Y = [40*e[3] for e in action_list]
plt.plot(smooth_curve(Y, 20), label="ouverture de la vanne d'air extérieur")
plt.plot(smooth_curve(T_out_list, 5), label="Température (°C)")

plt.legend()
plt.show()
