module ControlSystemsExt

using DyadLinearMPC
using ControlSystemsBase

function DyadLinearMPC.Model(sys::AbstractStateSpace;disturbance_ids = zeros(Int,0),Ts=1.0)
    A,B,C,D = ssdata(sys)
    control_ids = setdiff(1:size(B,2),disturbance_ids) 
    B,Bd = B[:,control_ids], B[:,disturbance_ids]
    Dd = D[:,disturbance_ids]
    iszero(D[:,control_ids]) || throw(ArgumentError("Non-proper system"))
    if isdiscrete(sys)
        return DyadLinearMPC.Model(A,B;C,Gd=Bd,Dd, Ts = ControlSystemsBase.timeevol(sys).Ts )
    else
        return DyadLinearMPC.Model(A,B,Ts;C,Bd,Dd)
    end
end

DyadLinearMPC.Model(sys::TransferFunction; Ts=1.0) = DyadLinearMPC.Model(ss(sys);Ts)
DyadLinearMPC.Model(sys::DelayLtiSystem; Ts=1.0) = DyadLinearMPC.Model(c2d(sys,Ts,:zoh);Ts)

end
