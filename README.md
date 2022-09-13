# Energetical-Optimisation-DQN-
This project aims to optimize the energy consumption of an air controlling device (humidity + temperature) in a room connected to the exterior by a thermal flow depending 
on its architecture (Windows, walls, doors, ...) by using a deep Q network.

Physical laws are simple heat transfers to the exterior using a flow modelisation taking into account an overall thermal resistance R of the room 
and its overall thermal capacity C (see the classical electrical/thermal ananlogy for further details).
The regulation device (see HVAC systems) both has an air intake on the outside and on the inside of the room, it can regulate the humidity and temperature passing through
the device, thus consuming energy.

Parameters of the air-controlling device are:
- amount of hot water used to warm the air (in [0, 1])
- amount of cold water used to cool the air (in [0, 1])
- amount of air taken from the inside (in [0, 1])
- amount of air taken from the outside (in [0, 1])
- humidificaton factor (in [0, 2])

These parameters are modeling the opening percentage of valves, as for real applications, and constitute the infinite state space of the simulation, 
whereas in most papers, they are discretized to match a classical Q network pattern.

Thus, the critical aspect of this project was the developpement of a Q Network that would be able to deal with an infinite state space as the used state variable are continuous.
This was achieved by using an optimum point research on a multivariable function represented by the neural network which gives as an output an reward estimation for the 
agent which is in charge of taking actions to solve the environment.

I found very little examples of such infinite state space deep Q networks, and I would gladly exchange with anyone having information about it or wanting further details
about my algorithm design.

