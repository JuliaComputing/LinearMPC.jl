# lb <= Au uk + Ax xk <= ub for k ∈ ks
# (additional terms Ar rₖ, Aw wₖ, Ad dₖ, Aup u⁻ₖ)

struct Constraint
    Au::Matrix{Float64}
    Ax::Matrix{Float64}
    Ar::Matrix{Float64}
    Aw::Matrix{Float64}
    Ad::Matrix{Float64}
    Aup::Matrix{Float64}
    Ap::Matrix{Float64}
    ub::Vector{Float64}
    lb::Vector{Float64}
    ks::AbstractVector{Int64}
    soft::Bool
    binary::Bool
    prio::Int
    tighten::Bool
end
Constraint(Au,Ax,Ar,Aw,Ad,Aup,Ap,ub,lb,ks,soft,binary,prio) =
    Constraint(Au,Ax,Ar,Aw,Ad,Aup,Ap,ub,lb,ks,soft,binary,prio,true)

# Weights used to define the objective function of the OCP
struct MPCWeights
    Q::Matrix{Float64}
    R::Matrix{Float64}
    Rr::Matrix{Float64}
    S::Matrix{Float64}
    Qf::Matrix{Float64}
    Qfx::Matrix{Float64}
    Ex::Matrix{Float64}
    ex::Vector{Float64}
    Eu::Matrix{Float64}
    eu::Vector{Float64}
    Sd::Matrix{Float64} # disturbance-control cross term d' Sd u (nd × nu); empty means zero
end

MPCWeights(Q,R,Rr,S,Qf,Qfx,Ex,ex,Eu,eu) =
    MPCWeights(Q,R,Rr,S,Qf,Qfx,Ex,ex,Eu,eu,zeros(0,size(R,1)))

function MPCWeights(nu,nx,nr)
    return MPCWeights(Matrix{Float64}(I,nr,nr),Matrix{Float64}(I,nu,nu),zeros(nu,nu),
                      zeros(nx,nu),zeros(nr,nr),zeros(nx,nx),zeros(nx,0),zeros(nx),zeros(nu,0),zeros(nu),zeros(0,nu))
end

function MPCWeights(Q::AbstractArray,R::AbstractArray,Rr::AbstractArray=zeros(size(R));
        S = zeros(0,0), Qf = zeros(0,0), Qfx = zeros(0,0),
        Ex = zeros(size(Q, 1), 0), ex = zeros(size(Q, 1)),
        Eu = zeros(size(R, 1), 0), eu = zeros(size(R, 1)), Sd = zeros(0,0))
    Qf = isempty(Qf) ? copy(Q) : Qf 
    return MPCWeights(matrixify(Q),matrixify(R),matrixify(Rr),float(S),matrixify(Qf),matrixify(Qfx),
                      float(Ex),float(ex),float(Eu),float(eu),matrixify(Sd))
end

"""
MPC controller settings.

# Fields
- `condensation_weights = zeros(0)`: weights used in the reference condensation
- `preprocess_mpqp::Bool = true`: Run preprocessing of mpqp to merge/remove constraints
- `reference_condensation::Bool = false`: Collapse reference trajectory to setpoint 
- `reference_tracking::Bool = true`: Enable reference tracking
- `reference_preview::Bool = false`: Enable time-varying reference preview
- `disturbance_preview::Bool = false`: Enable time-varying disturbance preview
- `parameter_preview::Bool = false`: Enable time-varying generalized-parameter preview
- `soft_weight::Float64 = 1e6`: Penalty weight for soft constraint violations
- `bnb_warm_start::Bool = false`: Warm start the branch and bound of problems with binary controls from the solution of the previous call (see [`solve`](@ref DyadLinearMPC.solve))
- `deferred_tol::Float64 = 0.0`: Largest increase of the objective over that of the relaxed search for which the resolved deferred binary controls are accepted without the full branch and bound (see [`defer_binary_controls!`](@ref))
- `deferred_max_combinations::Int = 64`: Largest number of combinations of the candidates of the groups of deferred binary controls that are evaluated (see [`defer_binary_controls!`](@ref))
- `branching::Int = 0`: Selection of the binary decision variable to branch over in the branch and bound of DAQP (its
  setting `branching`): the first (`0`) or the last (`1`) in the order of the decision variables, which is the order
  of the time steps, the most fractional (`2`), or the first in the order given by `branch_priority` (`3`)
- `branch_priority::Vector{Int} = Int[]`: Binary controls (indices of controls) in decreasing priority for
  `branching = 3`. The binary decision variables of a control are ordered by their time step, from the first, or from
  the last for a negative entry `-c` of the control `c`; those of the binary
  controls that are not listed follow in the order of the decision variables
- `bnb_portfolio::Vector{Int} = Int[]`: Branching rules (values of the setting `branching` of DAQP) of additional
  DAQP workspaces that solve each branch and bound together with the workspace of the MPC, each on a thread of its
  own (`DAQP.solve_parallel`): the threads share the best integer-feasible solution, and the solve ends when the
  first of them has explored its tree. The workspaces are set up with the MPC and keep their warm start between
  calls. Requires a version of DyadDAQP with `solve_parallel`; generated code solves with the workspace of the MPC only
- `bnb_split::Int = 0`: Number of binary decision variables of the first time step (that are not deferred, in the
  order of `bnb_split_controls`) on which each branch and bound is split (`0`: no split). For each combination of their
  values (their lower or upper bound), the branch and bound with these binary decision variables fixed is solved by
  DAQP workspaces of its own, one per branching rule (that of the workspace of the MPC, then those of
  `bnb_portfolio`), each on a thread of its own (`DAQP.solve_parallel` with `groups`, `2^bnb_split` times the
  number of rules). The threads share the best integer-feasible solution; the search of a combination is complete
  when one of its threads has explored its tree, and the solve is optimal when the searches of all combinations are
  complete. The workspace of the MPC searches the combination of the solution of the previous call (that of the
  candidate of `bnb_warm_start`), or that of the lower bounds if there is none. A binary decision variable whose
  bounds are equal has a single value, which halves the number of combinations. The workspaces are set up with
  the MPC and keep their warm start between calls. Requires a version of DyadDAQP with the keyword `groups` of
  `solve_parallel`; generated code solves with the workspace of the MPC only
- `bnb_split_controls::Vector{Int} = Int[]`: Binary controls (indices of controls) whose binary decision variables of
  the first time step are candidates for `bnb_split`, in this order (empty: all binary controls in their order)
- `solver_opts::Dict{Symbol,Any}`: Additional solver options
"""
Base.@kwdef mutable struct MPCSettings
    condensation_weights::Union{Vector{Float64},Matrix{Float64}}= zeros(0)
    preprocess_mpqp::Bool=true
    reference_condensation::Bool= false
    reference_tracking::Bool= true
    reference_preview::Bool = false
    disturbance_preview::Bool = false
    parameter_preview::Bool = false
    soft_weight::Float64= 1e6
    bnb_warm_start::Bool = false
    deferred_tol::Float64 = 0.0
    deferred_max_combinations::Int = 64
    branching::Int = 0
    branch_priority::Vector{Int} = Int[]
    bnb_portfolio::Vector{Int} = Int[]
    bnb_split::Int = 0
    bnb_split_controls::Vector{Int} = Int[]
    solver_opts::Dict{Symbol,Any} = Dict()
    traj2setpoint::Matrix{Float64} = zeros(0,0)
end

struct MPQP
    H::Matrix{Float64}
    f::Vector{Float64}
    H_theta::Matrix{Float64}
    f_theta::Matrix{Float64}

    A::Matrix{Float64}
    bu::Vector{Float64}
    bl::Vector{Float64}
    W::Matrix{Float64}

    senses::Vector{Cint}
    prio::Vector{Cint}
    break_points::Vector{Cint}

    has_binaries::Bool
    is_symmetric::Bool

    # Workspace arrays for solve() to avoid allocations
    _bth::Vector{Float64}
    _bu::Vector{Float64}
    _bl::Vector{Float64}
    _f::Vector{Float64}

    # Cholesky factor of H from a square root of the objective (nothing if not formed)
    Hchol::Union{Nothing,Cholesky{Float64,Matrix{Float64}}}
end

# (Without a factor of H)
MPQP(H,f,H_theta,f_theta,A,bu,bl,W,senses,prio,break_points,has_binaries,is_symmetric,_bth,_bu,_bl,_f) =
    MPQP(H,f,H_theta,f_theta,A,bu,bl,W,senses,prio,break_points,has_binaries,is_symmetric,_bth,_bu,_bl,_f,nothing)

function MPQP()
    return MPQP(Matrix{Float64}(undef, 0, 0),Float64[],Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0),
                Matrix{Float64}(undef, 0, 0),Float64[],Float64[], Matrix{Float64}(undef, 0, 0),
                Cint[],Cint[],Cint[],false,true,
                Float64[],Float64[],Float64[],Float64[],nothing)
end

# A group of binary controls that are resolved after the others (see defer_binary_controls!)
struct DeferredBinaryGroup
    ids::Vector{Int}          # Controls
    weights::Vector{Float64}  # Weights of the encoded integer (empty if there are none)
    resolution::Symbol        # :auto or :enumerate
end

# A general constraint lower ≤ ∑ coef[i]*x[binary_ids[pos[i]]] ≤ upper that involves only binary decision
# variables, with bounds that do not depend on the parameter (see bnb_logic_rows)
struct BnBLogicRow
    pos::Vector{Int}
    coef::Vector{Float64}
    lower::Float64
    upper::Float64
end

# The binary decision variables of a group of deferred binary controls (see setup_bnb!). Positions refer to
# BnBData.binary_ids.
struct BnBGroup
    kind::Symbol               # :encoding, :rounding (a single control) or :enumeration
    pos::Vector{Int}           # Positions of the binary decision variables (:encoding: step by step, each step in the order of the controls)
    weights::Vector{Float64}   # :encoding: the weights of the controls
    durations::Vector{Float64} # :rounding: the number of time steps of each binary decision variable
    rows::Vector{BnBLogicRow}  # :rounding: the logic constraints of the binary decision variables
end

# An additional DAQP workspace of the setting bnb_portfolio or bnb_split (see daqp_solve_bnb): its branching rule
# (-1 for that of the workspace of the MPC) and the senses that it was last updated with (mpQP.senses or
# BnBData.relaxed_senses)
mutable struct BnBPortfolioWorkspace
    model::DyadDAQP.Model
    branching::Int
    senses::Vector{Cint}
end

# Data of the solution strategies of the branch and bound (see solve_bnb). The binary decision variables are
# those whose simple bounds have the binary sense.
mutable struct BnBData
    binary_ids::Vector{Int} # Binary decision variables
    shift_ids::Vector{Int}  # The decision variable that holds the control of each binary decision variable one step later
    xprev::Vector{Float64}  # Solution of the previous call (empty if there is none)
    deferred::Vector{DeferredBinaryGroup} # Declared groups of deferred binary controls
    groups::Vector{BnBGroup}      # Their binary decision variables
    is_deferred::Vector{Bool}     # Whether each binary decision variable is deferred
    relaxed_senses::Vector{Cint}  # The senses of the constraints with the deferred binary decision variables relaxed
    portfolio::Vector{BnBPortfolioWorkspace} # Additional workspaces of the setting bnb_portfolio
    split_ids::Vector{Int}  # The binary decision variables of the setting bnb_split
    # For each combination of their values (bit k set: split_ids[k] at its upper bound), a workspace per branching
    # rule: that of the workspace of the MPC, then those of bnb_portfolio
    split::Vector{Vector{BnBPortfolioWorkspace}}
end
BnBData() = BnBData(Int[],Int[],Float64[],DeferredBinaryGroup[],BnBGroup[],Bool[],Cint[],BnBPortfolioWorkspace[],
                    Int[],Vector{BnBPortfolioWorkspace}[])

# MPC controller
mutable struct MPC

    model::Model

    # parameters
    nr::Int
    nd::Int
    nuprev::Int
    np::Int

    # Horizons 
    Np::Int # Prediction
    Nc::Int # Control

    ## 
    weights::MPCWeights

    # lb <= u <=ub
    umin::Vector{Float64}
    umax::Vector{Float64}
    binary_controls::Vector{Int64}
    Nc_binary::Union{Int,Vector{Int}}

    # General constraints 
    constraints::Vector{Constraint}

    # Settings
    settings::MPCSettings

    #Optimization problem
    mpQP::MPQP

    # DAQP optimization model
    opt_model::DyadDAQP.Model

    # Prestabilizing feedback
    K::Matrix{Float64}

    # Move blocks
    move_blocks::Vector{Vector{Int}}

    mpqp_issetup::Bool

    uprev::Vector{Float64}

    traj2setpoint::Matrix{Float64}

    state_observer

    Δx0::Vector{Float64}

    objectives::Vector{<:Tuple{MPCWeights,Vector{Int}}}

    # Solution strategies of the branch and bound
    bnb::BnBData
end

function MPC(model::Model;Np=10,Nc=Np)
    MPC(model,0,0,0,0,Np,Nc,
        MPCWeights(model.nu,model.nx,model.ny),
        zeros(0),zeros(0),zeros(0),-1,
        Constraint[],MPCSettings(),MPQP(),
        DAQP.Model(),zeros(model.nu,model.nx),Vector{Int}[],false, zeros(model.nu),zeros(0,0),
        nothing,zeros(model.nx),
        Tuple{MPCWeights,Vector{Int}}[],
        BnBData())
end

function MPC(F,G;Gd=zeros(0,0), C=zeros(0,0), Dd= zeros(0,0), f_offset=zeros(0), Ts= -1.0, Np=10, Nc = Np)
    MPC(Model(F,G;Gd,f_offset,C,Dd,Ts);Np,Nc);
end

function MPC(A,B,Ts::Float64; Bd = zeros(0,0), f_offset=zeros(0), C = zeros(0,0), Dd = zeros(0,0), Np=10, Nc=Np)
    MPC(Model(A,B,Ts;Bd,f_offset,C,Dd);Np,Nc)
end

function MPC(sys; Ts=1.0, Np=10, Nc=Np)
    MPC(Model(sys;Ts);Np,Nc)
end

struct ParameterRange
    xmin::Vector{Float64}
    xmax::Vector{Float64}

    rmin::Vector{Float64}
    rmax::Vector{Float64}

    dmin::Vector{Float64}
    dmax::Vector{Float64}

    umin::Vector{Float64}
    umax::Vector{Float64}

    pmin::Vector{Float64}
    pmax::Vector{Float64}
end


function ParameterRange(mpc::MPC)

    nx,nr,nd,nuprev,np = get_parameter_dims(mpc);

    xmin,xmax = -100*ones(nx),100*ones(nx)
    rmin,rmax = -100*ones(nr),100*ones(nr)
    dmin,dmax = -100*ones(nd),100*ones(nd)
    if(nuprev > 0)
        nmin,nmax = length(mpc.umin),length(mpc.umax)
        nb = max(nmin,nmax)
        umin = [mpc.umin;-100*ones(nb-nmin)]
        umax = [mpc.umax;+100*ones(nb-nmax)]
    else
        umin,umax = zeros(0),zeros(0)
    end
    pmin,pmax = -100*ones(np),100*ones(np)

    return ParameterRange(xmin,xmax,
                          rmin,rmax,
                          dmin,dmax,
                          umin,umax,
                          pmin,pmax)
end
