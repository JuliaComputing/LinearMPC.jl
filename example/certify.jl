using DyadLinearMPC
# Load inverted pendulum example and certify the iteration complexity
example = DyadLinearMPC.mpc_example("invpend")
mpc,range = example.mpc, example.range
result = DyadLinearMPC.certify(mpc;range)

using Plots
plot(result;parameters=[:x2,:u1p])
