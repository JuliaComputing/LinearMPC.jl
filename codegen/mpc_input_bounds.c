// Sets the bounds of the constraints that depend on the input bounds umin <= u <= umax
// (setting tunable_input_bounds). The entries of input_bound_ids index p = [umin; umax].
static c_float mpc_clamp_bound(c_float v){
    return v > MPC_BOUND_INF ? MPC_BOUND_INF : (v < -MPC_BOUND_INF ? -MPC_BOUND_INF : v);
}

void mpc_set_input_bounds(c_float* umin, c_float* umax){
    int i,id;
    c_float p;
    for(i = 0; i < N_INPUT_BOUND_UPPER; i++){
        id = input_bound_upper_ids[i];
        p = id < N_CONTROL ? umin[id] : umax[id-N_CONTROL];
        du[input_bound_upper_rows[i]] = mpc_clamp_bound(input_bound_upper_offset[i] + input_bound_upper_coef[i]*p);
    }
    for(i = 0; i < N_INPUT_BOUND_LOWER; i++){
        id = input_bound_lower_ids[i];
        p = id < N_CONTROL ? umin[id] : umax[id-N_CONTROL];
        dl[input_bound_lower_rows[i]] = mpc_clamp_bound(input_bound_lower_offset[i] + input_bound_lower_coef[i]*p);
    }
}
