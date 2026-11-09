using DyadLinearMPC, Test, LinearAlgebra

# An MPC whose control after update_input_bounds! is compared with that of an MPC set up with the same
# bounds by set_input_bounds!
function tib_mpc(; umin=nothing, umax=nothing, tunable=true, K=false, block=nothing, extra=false,
                 disturbance=false, binary=false)
    F = [1.0 0.1 0.0; 0.0 1.0 0.1; 0.0 0.0 0.9]
    G = [0.0 0.0; 0.005 0.0; 0.1 0.05]
    mpc = DyadLinearMPC.MPC(F, G; C = [1.0 0 0; 0 1 0], Np = 12, Nc = 8)
    set_objective!(mpc; Q = [1.0, 0.5], R = [0.1, 0.2], Rr = [0.05, 0.05])
    isnothing(umin) || set_input_bounds!(mpc; umin, umax)
    K && set_prestabilizing_feedback!(mpc)
    isnothing(block) || move_block!(mpc, block)
    if extra
        set_output_bounds!(mpc; ymin = [-4.0, -3.0], ymax = [4.0, 3.0])
        # A constraint on a single input, which the preprocessing merges into the simple bounds when they are fixed
        add_constraint!(mpc; Au = [0.0 1.0], lb = [-0.7], ub = [0.6], ks = 1:2)
    end
    disturbance && set_disturbance!(mpc, -0.001, 0.001)
    binary && set_binary_controls!(mpc, [2])
    settings!(mpc; tunable_input_bounds = tunable)
    setup!(mpc)
    return mpc
end

const TIB_X = [[2.0, -1.0, 0.5], [-3.0, 0.5, -1.0], [0.3, 0.2, 0.1]]

function tib_compare(kw, b0, b1)
    mpc = tib_mpc(; umin = b0[1], umax = b0[2], kw...)
    update_input_bounds!(mpc; umin = b1[1], umax = b1[2])
    ref = tib_mpc(; umin = b1[1], umax = b1[2], tunable = false, kw...)
    for x in TIB_X
        u = compute_control(mpc, x)
        uref = compute_control(ref, x)
        @test u ≈ uref atol = 1e-7
        isnothing(b1[1]) || @test all(b1[1] .- 1e-7 .<= u .<= b1[2] .+ 1e-7)
    end
    return mpc
end

@testset "Tunable input bounds" begin
    b0 = ([-1.0, -1.0], [1.0, 1.0])
    b1 = ([-0.3, -0.5], [0.2, 0.8])
    @testset "$name" for (name, kw) in (("simple bounds", (;)),
                                       ("move blocking", (; block = 2)),
                                       ("prestabilizing feedback", (; K = true)),
                                       ("output bounds and merged constraint", (; extra = true)),
                                       ("constraint tightening", (; K = true, disturbance = true)))
        tib_compare(kw, b0, b1)
        tib_compare(kw, b1, b0)
    end

    @testset "Same result as fixed bounds at the setup" begin
        for kw in ((;), (; K = true), (; extra = true))
            a = tib_mpc(; umin = b1[1], umax = b1[2], kw...)
            b = tib_mpc(; umin = b1[1], umax = b1[2], tunable = false, kw...)
            for x in TIB_X
                @test compute_control(a, x) ≈ compute_control(b, x) atol = 1e-7
            end
        end
    end

    @testset "Bounds absent at the setup" begin
        mpc = tib_mpc()
        ref = tib_mpc(; tunable = false)
        @test compute_control(mpc, TIB_X[2]; uprev = zeros(2)) ≈ compute_control(ref, TIB_X[2]; uprev = zeros(2)) atol = 1e-7
        tib_compare((;), (nothing, nothing), b1)
        # Removing the bounds again
        mpc = tib_mpc(; umin = b1[1], umax = b1[2])
        update_input_bounds!(mpc; umin = [-Inf, -Inf], umax = [Inf, Inf])
        @test compute_control(mpc, TIB_X[2]; uprev = zeros(2)) ≈ compute_control(ref, TIB_X[2]; uprev = zeros(2)) atol = 1e-7
        # Equal bounds
        mpc = tib_mpc(; umin = b0[1], umax = b0[2])
        update_input_bounds!(mpc; umin = [0.1, -0.5], umax = [0.1, 0.5])
        @test compute_control(mpc, TIB_X[1])[1] ≈ 0.1 atol = 1e-7
    end

    @testset "No change of the matrices" begin
        mpc = tib_mpc(; umin = b0[1], umax = b0[2])
        H, A = copy(mpc.mpQP.H), copy(mpc.mpQP.A)
        update_input_bounds!(mpc; umin = b1[1], umax = b1[2])
        @test mpc.mpqp_issetup
        @test mpc.mpQP.H == H && mpc.mpQP.A == A
        p = [b1[1]; b1[2]]
        DyadLinearMPC.apply_input_bounds!(mpc.mpQP, p)
        @test (@allocated DyadLinearMPC.apply_input_bounds!(mpc.mpQP, p)) == 0
    end

    @testset "Errors" begin
        mpc = tib_mpc(; umin = b0[1], umax = b0[2], tunable = false)
        @test_throws ArgumentError update_input_bounds!(mpc; umin = b1[1], umax = b1[2])
        mpc = tib_mpc(; umin = b0[1], umax = b0[2])
        @test_throws ArgumentError update_input_bounds!(mpc; umin = [0.5, 0.0], umax = [0.0, 0.0])
        @test_throws ArgumentError update_input_bounds!(mpc; umin = [0.0], umax = [1.0])
    end

    @testset "Binary controls" begin
        bb0 = ([-1.0, 0.0], [1.0, 1.0])
        bb1 = ([-0.3, 0.0], [0.2, 1.0])
        mpc = tib_compare((; binary = true), bb0, bb1)
        @test_throws ArgumentError update_input_bounds!(mpc; umin = [-0.3, -1.0], umax = [0.2, 1.0])
    end

    @testset "Codegen" begin
        if !isnothing(Sys.which("gcc"))
            for kw in ((;), (; K = true), (; block = 2))
                mpc = tib_mpc(; umin = b0[1], umax = b0[2], kw...)
                srcdir = tempname()
                DyadLinearMPC.codegen(mpc; dir = srcdir)
                src = [f for f in readdir(srcdir) if last(f, 1) == "c"]
                lib = "mpctib." * Base.Libc.Libdl.dlext
                run(Cmd(`gcc -lm -fPIC -O3 -xc -shared -o $lib $src`; dir = srcdir))
                h = Base.Libc.Libdl.dlopen(joinpath(srcdir, lib), Base.Libc.Libdl.RTLD_LOCAL)
                setb = Base.Libc.Libdl.dlsym(h, :mpc_set_input_bounds)
                comp = Base.Libc.Libdl.dlsym(h, :mpc_compute_control)
                for (bounds, ok) in ((b1, true), (([-Inf, -Inf], [Inf, Inf]), true), (b0, true))
                    ccall(setb, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}), bounds[1], bounds[2])
                    update_input_bounds!(mpc; umin = bounds[1], umax = bounds[2])
                    for x in TIB_X
                        u = zeros(2)
                        ccall(comp, Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
                              u, x, zeros(2), zeros(0))
                        @test u ≈ compute_control(mpc, x; uprev = zeros(2)) atol = 1e-6
                    end
                end
                Base.Libc.Libdl.dlclose(h)
            end
        end
    end
end
