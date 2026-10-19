using DyadLinearMPC

# Load an example and run its first predefined scenario
example = DyadLinearMPC.mpc_example("invpend")
sim = DyadLinearMPC.Simulation(example, 1)

## Plot the trajectory 
using Plots
plot(sim)
