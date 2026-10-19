using DyadLinearMPC
# Load inverted pendulum example and generate code 
example = DyadLinearMPC.mpc_example("invpend")
mpc,range = example.mpc, example.range
DyadLinearMPC.codegen(mpc;dir="codegen_impc")

# Compute explicit 
empc = DyadLinearMPC.ExplicitMPC(mpc;range)
DyadLinearMPC.codegen(empc;dir="codegen_empc")
