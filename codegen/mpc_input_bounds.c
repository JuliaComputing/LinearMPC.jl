// Changes the input bounds umin <= u <= umax (setting tunable_input_bounds). The bounds of the
// constraints that depend on them are formed in the coordinates of the QP, clamped to
// +-MPC_BOUND_INF and transformed to the coordinates of the LDP, as at the setup. The entries of
// input_bound_*_ids index p = [umin; umax].
// Returns 0, or (without a change) -1 if a bound is NaN or umin > umax, and -2 if a bound of a
// binary control differs from its value at the setup.
static c_float mpc_clamp_bound(c_float v){
    return v > MPC_BOUND_INF ? MPC_BOUND_INF : (v < -MPC_BOUND_INF ? -MPC_BOUND_INF : v);
}

// Marks constraint i as an equality constraint if its bounds are equal, as DAQP does when the bounds
// are updated (daqp_check_bounds)
static void mpc_mark_input_bound_equality(int i){
    DAQPWorkspace* work = &daqp_work;
    if(du[i]-dl[i] < work->settings->zero_tol && !DAQP_IS_SOFT(i) && !DAQP_IS_BINARY(i))
        work->sense[i] |= DAQP_ACTIVE | DAQP_IMMUTABLE | DAQP_AUTO_EQUALITY;
    else if(work->sense[i] & DAQP_AUTO_EQUALITY)
        work->sense[i] &= ~(DAQP_ACTIVE | DAQP_IMMUTABLE | DAQP_AUTO_EQUALITY);
}

int mpc_set_input_bounds(c_float* umin, c_float* umax){
    int i,id;
    c_float p;
    for(i = 0; i < N_CONTROL; i++)
        if(umin[i] != umin[i] || umax[i] != umax[i] || umin[i] > umax[i]) return -1;
    for(i = 0; i < N_INPUT_BOUND_FIXED; i++){
        id = input_bound_fixed_ids[i];
        if(umin[id] != input_bound_fixed_lower[i] || umax[id] != input_bound_fixed_upper[i]) return -2;
    }
    // The working set, which may hold equality constraints of the previous bounds, is formed
    // again by the next solve
    daqp_deactivate_constraints(&daqp_work);
    for(i = 0; i < N_INPUT_BOUND_UPPER; i++){
        id = input_bound_upper_ids[i];
        p = id < N_CONTROL ? umin[id] : umax[id-N_CONTROL];
        du[input_bound_upper_rows[i]] = (mpc_clamp_bound(input_bound_upper_offset[i] + input_bound_upper_coef[i]*p)
                                         + input_bound_upper_shift[i])*input_bound_upper_invscale[i];
    }
    for(i = 0; i < N_INPUT_BOUND_LOWER; i++){
        id = input_bound_lower_ids[i];
        p = id < N_CONTROL ? umin[id] : umax[id-N_CONTROL];
        dl[input_bound_lower_rows[i]] = (mpc_clamp_bound(input_bound_lower_offset[i] + input_bound_lower_coef[i]*p)
                                         + input_bound_lower_shift[i])*input_bound_lower_invscale[i];
    }
    for(i = 0; i < N_INPUT_BOUND_UPPER; i++) mpc_mark_input_bound_equality(input_bound_upper_rows[i]);
    for(i = 0; i < N_INPUT_BOUND_LOWER; i++) mpc_mark_input_bound_equality(input_bound_lower_rows[i]);
    return 0;
}
