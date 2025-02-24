using Distributed
using HypothesisTests
using GLM
using EffectSizes
using Bootstrap
using CategoricalArrays
using SimpleANOVA
using CairoMakie
using ColorSchemes
CairoMakie.activate!()

addprocs(...) # set number of cores for parallel processing

@everywhere begin
    using CSV
    using Random
    using Combinatorics
    using Colors
    using DataFrames
    using Distributions
    using LinearAlgebra
    using Distances
    using Clustering
    using ClusterAnalysis
    using StatsBase
    using MultivariateStats
    using Flux
end

palette = ColorSchemes.Set2_4;

@everywhere function luv_convert(crds, i)
    c = convert(Luv, RGB(crds[i, :]...))
    return c.l, c.u, c.v
end

#= to run the study in CIELAB, everywhere use this function instead of the one above:
@everywhere function lab_convert(crds, i)
    c = convert(Lab, RGB(crds[i, :]...))
    return c.l, c.a, c.b
end
=#

@everywhere coords_full = CSV.read("/.../munsell_rgb.csv", DataFrame) |> Matrix
@everywhere luv_coords_full = [ luv_convert(coords_full, i) for i in 1:size(coords_full, 1) ]

@everywhere coords = CSV.read("/.../rgb320.csv", DataFrame; header=false)./255 |> Matrix
@everywhere luv_coords = [ luv_convert(coords, i) for i in 1:size(coords, 1) ]

@everywhere luv_df = DataFrame(luv_coords)

# assume 10 clusters (given that gray is often not represented in the naming data for the 320 chromatic WCS chips)
@everywhere label(v::Vector{Vector{Float64}}) = [ findmin([ Distances.evaluate(Euclidean(), v[i], luv_df[j, :]) for i in 1:10 ])[2] for j in axes(luv_df, 1) ]

@everywhere ca = ClusterAnalysis.kmeans(luv_df, 10; nstart=250, maxiter=100)

@everywhere cl = [ vcat(collect.([luv_coords_full[sample(1:1625, 10; replace=false)]...])) for _ in 1:500 ]
@everywhere rand_clust = label.(cl)
# sometimes the above procedure will not result in 10 categories, so we create 500 random clusterings as needed and select 100 with the required number of clusters
@everywhere random_clustering = rand_clust[length.(unique.(rand_clust)) .== 10][1:100]
# the following gives 100 sets of 10 color prototypes
for_calc = cl[length.(unique.(rand_clust)) .== 10][1:100]

# helper functions

bs(x; n=1000) = bootstrap(mean, x, BasicSampling(n)) # bootstrap sampling

@everywhere function compute_aulc(accuracies::Vector{Float64}) # compute AULC
    n = length(accuracies)
    area = sum(0.5 * (accuracies[i] + accuracies[i+1]) for i in 1:(n-1))
    return area
end

# define constants

@everywhere const df_xs = reduce(hcat, convert.(Vector{Float32}, collect.(luv_coords)))
@everywhere const numb_nets = 50

# make data for ANNs

@everywhere function make_data(labs::Vector{Int64})
    df = DataFrame(luv_coords)
    df.lbs = labs
    df_gb = groupby(df, :lbs)
    df_samp = reduce(vcat, [ df_gb[i][sample(axes(df_gb[i], 1), rand(axes(df_gb[i], 1)); replace=false, ordered=true), :] for i in 1:10 ])
    m_samp = Matrix(df_samp)
    xs = Matrix{Float32}(m_samp[:, 1:3]')
    ys = Flux.onehotbatch(Int.(m_samp[:, 4]), 1:10)
    return Flux.DataLoader((xs, ys), batchsize=3, shuffle=true)
end

# to use the natural clustering for labeling, we run
make_data(ca.cluster)
# to let it generate data on the basis of some nonnatural
# system, we use, e.g.,
make_data(random_clustering[1])

@everywhere begin
    flatten_params(mlp) = first(Flux.destructure(mlp))

    function cosine_similarity(vec1, vec2)
        dot_product = dot(vec1, vec2)
        norm_vec1 = norm(vec1)
        norm_vec2 = norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)
    end

    function params_sim(x)
        fl = flatten_params.(x)
        return [ cosine_similarity(fl[i], fl[j]) for i in 1:numb_nets, j in 1:numb_nets ]
    end

    function params_sim(x, n)
        fl = flatten_params.(x)
        return [ cosine_similarity(fl[i], fl[j]) for i in 1:n, j in 1:n ]
    end
end

# state-based updating
@everywhere function sb_updating_for_nets(nets::Vector{Chain{Tuple{Dense{typeof(relu), Matrix{Float32}, Vector{Float32}}, Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}}}},
                                          ϵ::Float32,
                                          α::Float32,
                                          labs::Vector{Int64})

    # find peers and store average parameters (no social updating yet)
    peer_dist = params_sim(nets)
    p_mat = peer_dist .≥ 1 - ϵ
    p_means = Vector{VecOrMat{Float32}}[]
    for n in 1:numb_nets
        p = findall(==(1), p_mat[n, :])
        v = VecOrMat{Float32}[]
        for i in 1:4
            m = mean([ Flux.trainables(nets[j])[i] for j in p ], dims=1)[1]
            push!(v, m)
        end
        push!(p_means, v)
    end

    # worldly part of updating
    data = [ make_data(labs) for _ in 1:numb_nets ]
    for i in 1:numb_nets
        f, re = Flux.destructure(nets[i])
        opt = Flux.setup(Adam(), f)
        Flux.train!((p, x, y) -> Flux.logitcrossentropy(re(p)(x), y), f, data[i], opt)
        nets[i] = re(f)
    end

    # hk updating
    for n in 1:numb_nets
        for i in 1:4
            m = α .* Flux.trainables(nets[n])[i] .+ (1. - α) .* p_means[n][i]
            Flux.trainables(nets[n])[i] .= m
        end
    end

    return [ Flux.onecold(model(df_xs), 1:10) for model ∈ nets ]
end

@everywhere function state_based_natural(; ϵ::Float32=.9f0, α::Float32=.9f0, numb_epochs::Int=100)
    nets = [ Flux.Chain(Dense(3, 9, Flux.relu), Dense(9, 10)) for _ in 1:numb_nets ]
    res = [ sb_updating_for_nets(nets, ϵ, α, ca.cluster) for _ in 1:numb_epochs ]
    return res
end

@everywhere function state_based_non_natural(labs::Vector{Int64}; ϵ::Float32=.9f0, α::Float32=.9f0, numb_epochs::Int=100)
    nets = [ Flux.Chain(Dense(3, 9, Flux.relu), Dense(9, 10)) for _ in 1:numb_nets ]
    res = [ sb_updating_for_nets(nets, ϵ, α, labs) for _ in 1:numb_epochs ]
    return res
end

sim_nat = pmap(_->state_based_natural(), 1:25)
sim_non_nat = pmap(i->state_based_non_natural(random_clustering[i]), 1:100)

mn = [ mean([ mutualinfo(ca.cluster, sim_nat[k][j][i]) for i in 1:numb_nets ]) for j in 1:100, k in 1:25 ]
mnn = [ mean([ mutualinfo(random_clustering[k], sim_non_nat[k][j][i]) for i in 1:numb_nets ]) for j in 1:100, k in 1:100 ]

state_based_bs_nat = [ Bootstrap.confint(bs(mn[i, :]), BasicConfInt(.95))[1] for i in axes(mn, 1) ]
state_based_bs_non_nat = [ Bootstrap.confint(bs(mnn[i, :]), BasicConfInt(.95))[1] for i in axes(mnn, 1) ]

pvals = [ pvalue(EqualVarianceTTest(mn[i, :], mnn[i, :])) for i in axes(mn, 1) ]

cohend = [ effectsize(CohenD(mn[i, :], mnn[i, :])) for i in axes(mn, 1) ]

f = Figure(fontsize=15, size=(780, 510));
ax1 = Axis(f[1, 1], xlabel="Epoch", xlabelsize=16, ylabel="Mean NMI", ylabelsize=16, title="State-based social updating", titlesize=22)
ax2 = Axis(f[1, 1], ylabel=rich("d"; font=:italic), ylabelsize=16, yaxisposition=:right)
l1 = lines!(ax1, 1:size(mn, 1), first.(state_based_bs_nat), color=:lightsteelblue4)
b1 = band!(ax1, 1:size(mn, 1), getindex.(state_based_bs_nat, 2), last.(state_based_bs_nat), color=(:lightsteelblue4, .5))
l2 = lines!(ax1, 1:size(mnn, 1), first.(state_based_bs_non_nat), color=:goldenrod)
b2 = band!(ax1, 1:size(mnn, 1), getindex.(state_based_bs_non_nat, 2), last.(state_based_bs_non_nat), color=(:goldenrod, .5))
l3 = scatter!(ax2, 1:size(mnn, 1), cohend, markersize=7, color=[:indianred3, :olivedrab4][Int.(pvals .< .01) .+ 1])
elem_1 = MarkerElement(color=:olivedrab4, marker = '⚫', markersize=18, points=Point2f[(.5, .5)])
elem_2 = MarkerElement(color=:indianred3, marker = '⚫', markersize=18, points=Point2f[(.5, .5)])
Legend(f[2, 1], [[l1, b1], [l2, b2], elem_1, elem_2], ["Natural", "Nonnatural", rich("Cohen's ", rich("d"; font=:italic), " (", rich("p"; font=:italic), " < .01)"), rich("Cohen's ", rich("d"; font=:italic), " (", rich("p"; font=:italic), " ≥ .01)")], framevisible=false, orientation = :horizontal, tellwidth = false, tellheight = true)
f

nat_aulc = reduce(vcat, [ mapslices(compute_aulc, [ mutualinfo(ca.cluster, sim_nat[k][j][i]) for i in 1:numb_nets, j in 1:100 ], dims=2) for k in 1:25 ])
non_nat_aulc = reduce(vcat, [ mapslices(compute_aulc, [ mutualinfo(random_clustering[k], sim_non_nat[k][j][i]) for i in 1:numb_nets, j in 1:100 ], dims=2) for k in 1:100 ])

n_cond = vcat(fill(1, 1250), fill(2, 5000))
fig_bp = Figure(fontsize=15, size = (588, 396));
axbp = Axis(fig_bp[1,1]; ylabel="AULC", xlabelsize=16, ylabelsize=16, xticks=(1:2, ["Natural", "Nonnatural"]), title="State-based social updating", titlesize=22)
boxplot!(axbp, n_cond, dropdims(vcat(nat_aulc, non_nat_aulc), dims=2), whiskerwidth = .5, width = 0.25, gap=-1.5, show_notch = false, show_outliers=true, color=(:lightsteelblue4, .5))
fig_bp

# individual updating

@everywhere function nonsocial_updating_for_nets(nets::Vector{Chain{Tuple{Dense{typeof(relu), Matrix{Float32}, Vector{Float32}}, Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}}}}, labs::Vector{Int64})
    data = [ make_data(labs) for _ in 1:numb_nets ]
    for i in 1:numb_nets
        f, re = Flux.destructure(nets[i])
        opt = Flux.setup(Adam(), f)
        Flux.train!((p, x, y) -> Flux.logitcrossentropy(re(p)(x), y), f, data[i], opt)
        nets[i] = re(f)
    end
    return [ Flux.onecold(model(df_xs), 1:10) for model ∈ nets ]
end

@everywhere function nonsocial_natural(; numb_epochs::Int=100)
    nets = [ Flux.Chain(Dense(3, 9, Flux.relu), Dense(9, 10)) for _ in 1:numb_nets ]
    res = [ nonsocial_updating_for_nets(nets, ca.cluster) for _ in 1:numb_epochs ]
    return res
end

@everywhere function nonsocial_non_natural(labs::Vector{Int64}; numb_epochs::Int=100)
    nets = [ Flux.Chain(Dense(3, 9, Flux.relu), Dense(9, 10)) for _ in 1:numb_nets ]
    res = [ nonsocial_updating_for_nets(nets, labs) for _ in 1:numb_epochs ]
    return res
end

sim_nat_ind = pmap(_->nonsocial_natural(), 1:25)
sim_non_nat_ind = pmap(i->nonsocial_non_natural(random_clustering[i]), 1:100)

mn_ind = [ mean([ mutualinfo(ca.cluster, sim_nat_ind[k][j][i]) for i in 1:numb_nets ]) for j in 1:100, k in 1:25 ]
mnn_ind = [ mean([ mutualinfo(random_clustering[k], sim_non_nat_ind[k][j][i]) for i in 1:numb_nets ]) for j in 1:100, k in 1:100 ]

ind_bs_nat = [ Bootstrap.confint(bs(mn_ind[i, :]), BasicConfInt(.95))[1] for i in axes(mn_ind, 1) ]
ind_bs_non_nat = [ Bootstrap.confint(bs(mnn_ind[i, :]), BasicConfInt(.95))[1] for i in axes(mnn_ind, 1) ]

pvals_ind = [ pvalue(EqualVarianceTTest(mn_ind[i, :], mnn_ind[i, :])) for i in axes(mn_ind, 1) ]

cohend_ind = [ effectsize(CohenD(mn_ind[i, :], mnn_ind[i, :])) for i in axes(mn_ind, 1) ]

fi = Figure(fontsize=15, size=(780, 510));
ax1i = Axis(fi[1, 1], xlabel="Epoch", xlabelsize=16, ylabel="Mean NMI", ylabelsize=16, title="Nonsocial updating", titlesize=22)
ax2i = Axis(fi[1, 1], ylabel=rich("d"; font=:italic), ylabelsize=16, yaxisposition=:right)
l1i = lines!(ax1i, 1:size(mn_ind, 1), first.(ind_bs_nat), color=:lightsteelblue4)
b1i = band!(ax1i, 1:size(mn_ind, 1), getindex.(ind_bs_nat, 2), last.(ind_bs_nat), color=(:lightsteelblue4, .5))
l2i = lines!(ax1i, 1:size(mnn_ind, 1), first.(ind_bs_non_nat), color=:goldenrod)
b2i = band!(ax1i, 1:size(mnn_ind, 1), getindex.(ind_bs_non_nat, 2), last.(ind_bs_non_nat), color=(:goldenrod, .5))
l3i = scatter!(ax2i, 1:size(mnn_ind, 1), cohend_ind, markersize=7, color=[:indianred3, :olivedrab4][Int.(pvals_ind .< .01) .+ 1])
elem_1 = MarkerElement(color=:olivedrab4, marker = '⚫', markersize=18, points=Point2f[(.5, .5)])
elem_2 = MarkerElement(color=:indianred3, marker = '⚫', markersize=18, points=Point2f[(.5, .5)])
Legend(fi[2, 1], [[l1i, b1i], [l2i, b2i], elem_1, elem_2], ["Natural", "Nonnatural", rich("Cohen's ", rich("d"; font=:italic), " (", rich("p"; font=:italic), " < .01)"), rich("Cohen's ", rich("d"; font=:italic), " (", rich("p"; font=:italic), " ≥ .01)")], framevisible=false, orientation = :horizontal, tellwidth = false, tellheight = true)
fi

# output-based updating
@everywhere function ob_updating_for_nets(nets::Vector{Chain{Tuple{Dense{typeof(relu), Matrix{Float32}, Vector{Float32}}, Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}}}},
                                          ϵ::Float32,
                                          α::Float32,
                                          labs::Vector{Int64})

    # worldly part of updating
    data = [ make_data(labs) for _ in 1:numb_nets ]
    for i in 1:numb_nets
        f, re = Flux.destructure(nets[i])
        opt = Flux.setup(Adam(), f)
        Flux.train!((p, x, y) -> Flux.logitcrossentropy(re(p)(x), y), f, data[i], opt)
        nets[i] = re(f)
    end

    # best guess clusterings at this point
    best_guess = [ Flux.onecold(model(df_xs), 1:10) for model ∈ nets ]

    # find peers
    guess_sim = [ mutualinfo(best_guess[i], best_guess[j]) for i in 1:numb_nets, j in 1:numb_nets ]
    p_mat = guess_sim .≥ 1 - ϵ
    p_mat[diagind(p_mat)] .= 0

    # take, per net, modal response for each chip
    modes = Vector{Int64}[]
    for j in 1:numb_nets
        bg = Vector{Int64}[]
        push!(bg, best_guess[j])
        bg_peers = best_guess[p_mat[j, :]]
        append!(bg, bg_peers)
        l = ceil(Int, α * length(bg_peers))
        append!(bg, fill(best_guess[j], l))
        v = [ mode(getindex.(bg, i)) for i in 1:320 ]
        push!(modes, v)
    end

    return modes
end

@everywhere function outcome_based_natural(; ϵ::Float32=.3f0, α::Float32=.1f0, numb_epochs::Int=100)
    nets = [ Flux.Chain(Dense(3, 9, Flux.relu), Dense(9, 10)) for _ in 1:numb_nets ]
    res = [ ob_updating_for_nets(nets, ϵ, α, ca.cluster) for _ in 1:numb_epochs ]
    return res
end

@everywhere function outcome_based_non_natural(labs::Vector{Int64}; ϵ::Float32=.3f0, α::Float32=.1f0, numb_epochs::Int=100)
    nets = [ Flux.Chain(Dense(3, 9, Flux.relu), Dense(9, 10)) for _ in 1:numb_nets ]
    res = [ ob_updating_for_nets(nets, ϵ, α, labs) for _ in 1:numb_epochs ]
    return res
end

sim_nat_ob = pmap(_->outcome_based_natural(), 1:25)
sim_non_nat_ob = pmap(i->outcome_based_non_natural(random_clustering[i]), 1:100)

mn_ob = [ mean([ mutualinfo(ca.cluster, sim_nat_ob[k][j][i]) for i in 1:numb_nets ]) for j in 1:100, k in 1:25 ]
mnn_ob = [ mean([ mutualinfo(random_clustering[k], sim_non_nat_ob[k][j][i]) for i in 1:numb_nets ]) for j in 1:100, k in 1:100 ]

outcome_based_bs_nat = [ Bootstrap.confint(bs(mn_ob[i, :]), BasicConfInt(.95))[1] for i in axes(mn_ob, 1) ]
outcome_based_bs_non_nat = [ Bootstrap.confint(bs(mnn_ob[i, :]), BasicConfInt(.95))[1] for i in axes(mnn_ob, 1) ]

pvals_ob = [ pvalue(EqualVarianceTTest(mn_ob[i, :], mnn_ob[i, :])) for i in axes(mn_ob, 1) ]

cohend_ob = [ effectsize(CohenD(mn_ob[i, :], mnn_ob[i, :])) for i in axes(mn_ob, 1) ]

fo = Figure(fontsize=15, size=(780, 510));
ax1o = Axis(fo[1, 1], xlabel="Epoch", xlabelsize=16, ylabel="Mean NMI", ylabelsize=16, title="Output-based social updating", titlesize=22)
ax2o = Axis(fo[1, 1], ylabel=rich("d"; font=:italic), ylabelsize=16, yaxisposition=:right)
l1o = lines!(ax1o, 1:size(mn_ob, 1), first.(outcome_based_bs_nat), color=:lightsteelblue4)
b1o = band!(ax1o, 1:size(mn_ob, 1), getindex.(outcome_based_bs_nat, 2), last.(outcome_based_bs_nat), color=(:lightsteelblue4, .5))
l2o = lines!(ax1o, 1:size(mnn_ob, 1), first.(outcome_based_bs_non_nat), color=:goldenrod)
b2o = band!(ax1o, 1:size(mnn_ob, 1), getindex.(outcome_based_bs_non_nat, 2), last.(outcome_based_bs_non_nat), color=(:goldenrod, .5))
l3o = scatter!(ax2o, 1:size(mnn_ob, 1), cohend_ob, markersize=7, color=[:indianred3, :olivedrab4][Int.(pvals .< .01) .+ 1])
Legend(fo[2, 1], [[l1o, b1o], [l2o, b2o], elem_1, elem_2], ["Natural", "Nonnatural", rich("Cohen's ", rich("d"; font=:italic), " (", rich("p"; font=:italic), " < .01)"), rich("Cohen's ", rich("d"; font=:italic), " (", rich("p"; font=:italic), " ≥ .01)")], framevisible=false, orientation = :horizontal, tellwidth = false, tellheight = true)
fo

ff = Figure(fontsize=15, size=(780, 510));
ax1ff = Axis(ff[1, 1], xlabel="Epoch", xlabelsize=16, ylabel="Mean NMI", ylabelsize=16, title="Social vs nonsocial updating/natural vs nonnatural concepts", titlesize=22)
#ax2ff = Axis(ff[1, 1], ylabel=rich("d"; font=:italic), ylabelsize=16, yaxisposition=:right)
l1ff = lines!(ax1ff, 1:size(mn_ob, 1), first.(outcome_based_bs_nat), color=palette[1])
b1ff = band!(ax1ff, 1:size(mn_ob, 1), getindex.(outcome_based_bs_nat, 2), last.(outcome_based_bs_nat), color=(palette[1], .5))
l2ff = lines!(ax1ff, 1:size(mnn_ob, 1), first.(outcome_based_bs_non_nat), color=palette[2])
b2ff = band!(ax1ff, 1:size(mnn_ob, 1), getindex.(outcome_based_bs_non_nat, 2), last.(outcome_based_bs_non_nat), color=(palette[2], .5))
l1i = lines!(ax1ff, 1:size(mn_ind, 1), first.(ind_bs_nat), color=palette[3])
b1i = band!(ax1ff, 1:size(mn_ind, 1), getindex.(ind_bs_nat, 2), last.(ind_bs_nat), color=(palette[3], .5))
l2i = lines!(ax1ff, 1:size(mnn_ind, 1), first.(ind_bs_non_nat), color=palette[4])
b2i = band!(ax1ff, 1:size(mnn_ind, 1), getindex.(ind_bs_non_nat, 2), last.(ind_bs_non_nat), color=(palette[4], .5))
#l3ff = scatter!(ax2o, 1:size(mnn_ob, 1), cohend_ob, markersize=7, color=[:indianred3, :olivedrab4][Int.(pvals .< .01) .+ 1])
Legend(ff[2, 1], [[l1ff, b1ff], [l2ff, b2ff], [l1i, b1i], [l2i, b2i]], ["Natural", "Nonnatural", rich("Cohen's ", rich("d"; font=:italic), " (", rich("p"; font=:italic), " < .01)"), rich("Cohen's ", rich("d"; font=:italic), " (", rich("p"; font=:italic), " ≥ .01)")], framevisible=false, orientation = :horizontal, tellwidth = false, tellheight = true)
ff

ff = Figure(fontsize=16, size=(770, 525));
ax1ff = Axis(ff[1, 1], xlabel="Epoch", xlabelsize=16, ylabel="Mean NMI", ylabelsize=16, title="Social vs nonsocial updating × natural vs nonnatural concepts", titlesize=22)
l1ff = lines!(ax1ff, 1:size(mn_ob, 1), first.(outcome_based_bs_nat), color=palette[1], linewidth=2)
l1l = lines!(ax1ff, 1:size(mn_ob, 1), getindex.(outcome_based_bs_nat, 2), color=palette[1], linestyle=:dot)
l1u = lines!(ax1ff, 1:size(mn_ob, 1), last.(outcome_based_bs_nat), color=palette[1], linestyle=:dot)
l2ff = lines!(ax1ff, 1:size(mnn_ob, 1), first.(outcome_based_bs_non_nat), color=palette[2], linewidth=2)
l2l = lines!(ax1ff, 1:size(mnn_ob, 1), getindex.(outcome_based_bs_non_nat, 2), color=palette[2], linestyle=:dot)
l2u = lines!(ax1ff, 1:size(mnn_ob, 1), last.(outcome_based_bs_non_nat), color=palette[2], linestyle=:dot)
l1i = lines!(ax1ff, 1:size(mn_ind, 1), first.(ind_bs_nat), color=palette[1], linewidth=2, linestyle=:dash)
l1il = lines!(ax1ff, 1:size(mn_ind, 1), getindex.(ind_bs_nat, 2), color=palette[1], linestyle=:dot)
l1iu = lines!(ax1ff, 1:size(mn_ind, 1), last.(ind_bs_nat), color=palette[1], linestyle=:dot)
l2i = lines!(ax1ff, 1:size(mnn_ind, 1), first.(ind_bs_non_nat), color=palette[2], linewidth=2, linestyle=:dash)
l2il = lines!(ax1ff, 1:size(mnn_ind, 1), getindex.(ind_bs_non_nat, 2), color=palette[2], linestyle=:dot)
l2iu = lines!(ax1ff, 1:size(mnn_ind, 1), last.(ind_bs_non_nat), color=palette[2], linestyle=:dot)
elem_1 = MarkerElement(color=palette[1], marker = '■', markersize=17, points=Point2f[(.5, .5)])
elem_2 = MarkerElement(color=palette[2], marker = '■', markersize=17, points=Point2f[(.5, .5)])
l3ff = lines!(ax1ff, 1:2, [0, 0], color=:black, linewidth=2)
l3ffl = lines!(ax1ff, 1:2, [0, 0], color=:black, linestyle=:dash)
l3ffu = lines!(ax1ff, 1:2, [0, 0], color=:black, linestyle=:dash)
l4ff = lines!(ax1ff, 1:2, [0, 0], color=:black, linewidth=2, linestyle=:dash)
l4ffl = lines!(ax1ff, 1:2, [0, 0], color=:black, linestyle=:dash)
l4ffu = lines!(ax1ff, 1:2, [0, 0], color=:black, linestyle=:dash)
Legend(ff[2, 1], [elem_1, elem_2, [l3ff, l3ffl, l3ffu], [l4ff, l4ffl, l4ffu]], ["Natural", "Nonnatural", "Social", "Nonsocial"], framevisible=false, orientation = :horizontal, tellwidth = false, tellheight = true)
ylims!(ax1ff, .3, .9)
ff

pvals_cross = [ pvalue(EqualVarianceTTest(mn_ob[i, :], mn_ind[i, :])) for i in axes(mn_ob, 1) ]
cohend_cross = [ effectsize(CohenD(mn_ob[i, :], mn_ind[i, :])) for i in axes(mn_ob, 1) ]
pvals_cross_nn = [ pvalue(EqualVarianceTTest(mnn_ob[i, :], mnn_ind[i, :])) for i in axes(mnn_ob, 1) ]
cohend_cross_nn = [ effectsize(CohenD(mnn_ob[i, :], mnn_ind[i, :])) for i in axes(mnn_ob, 1) ]

fc = Figure(fontsize=16, size=(770, 525));
ax1fc = Axis(fc[1, 1], xlabel="Epoch", xlabelsize=16, ylabel=rich("Cohen's ", rich("d"; font=:italic)), ylabelsize=16, title="Effect of social updating", titlesize=22)
l1c = scatter!(ax1fc, 1:size(mn_ob, 1), cohend_cross, marker = :diamond, color=[:indianred3, :olivedrab4][Int.(pvals_cross .< .01) .+ 1])
l2c = scatter!(ax1fc, 1:size(mnn_ob, 1), cohend_cross_nn, marker = :circle, color=[:indianred3, :olivedrab4][Int.(pvals_cross_nn .< .01) .+ 1])
elem_1 = MarkerElement(color=:black, marker = :diamond, markersize=17, points=Point2f[(.5, .5)])
elem_2 = MarkerElement(color=:black, marker = :circle, markersize=17, points=Point2f[(.5, .5)])
elem_3 = MarkerElement(color=:olivedrab4, marker = :rect, markersize=18, points=Point2f[(.5, .5)])
elem_4 = MarkerElement(color=:indianred3, marker = :rect, markersize=18, points=Point2f[(.5, .5)])
Legend(fc[2, 1], [elem_1, elem_2, elem_3, elem_4], ["Natural", "Nonnatural", rich(rich("p"; font=:italic), " < .01"), rich(rich("p"; font=:italic), " ≥ .01")], framevisible=false, orientation = :horizontal, tellwidth = false, tellheight = true)
ylims!(ax1fc, -1, 9)
fc

nat_aulc_ob = reduce(vcat, [ mapslices(compute_aulc, [ mutualinfo(ca.cluster, sim_nat_ob[k][j][i]) for i in 1:numb_nets, j in 1:100 ], dims=2) for k in 1:25 ])
non_nat_aulc_ob = reduce(vcat, [ mapslices(compute_aulc, [ mutualinfo(random_clustering[k], sim_non_nat_ob[k][j][i]) for i in 1:numb_nets, j in 1:100 ], dims=2) for k in 1:100 ])

nat_aulc_ind = reduce(vcat, [ mapslices(compute_aulc, [ mutualinfo(ca.cluster, sim_nat_ind[k][j][i]) for i in 1:numb_nets, j in 1:100 ], dims=2) for k in 1:25 ])
non_nat_aulc_ind = reduce(vcat, [ mapslices(compute_aulc, [ mutualinfo(random_clustering[k], sim_non_nat_ind[k][j][i]) for i in 1:numb_nets, j in 1:100 ], dims=2) for k in 1:100 ])

n_cond = vcat(fill(1, 1250), fill(2, 5000), fill(3, 1250), fill(4, 5000))
fig_bp_ob = Figure(fontsize=15, size = (748, 396));
axbp_ob = Axis(fig_bp_ob[1,1]; ylabel="AULC", xlabelsize=16, ylabelsize=16, xticks=(1:4, ["Social + natural", "Social + nonnatural", "Individual + natural", "Individual + nonnatural"]), title="Social vs nonsocial updating × natural vs nonnatural concepts: AULC", titlesize=22)
boxplot!(axbp_ob, n_cond, dropdims(vcat(nat_aulc_ob, non_nat_aulc_ob, nat_aulc_ind, non_nat_aulc_ind), dims=2), color=(:cornflowerblue, .6), whiskerwidth = .2, whiskerlinewidth=2, width = .8, whiskercolor=:cornflowerblue, strokecolor=:cornflowerblue, strokewidth=2, mediancolor=:cornflowerblue, medianlinewidth=2.2)
ylims!(axbp_ob,32, 89)
fig_bp_ob

# two-way ANOVA with interaction term
df_aov = DataFrame(
    outcome = dropdims(vcat(nat_aulc_ob, non_nat_aulc_ob, nat_aulc_ind, non_nat_aulc_ind), dims=2),
    concept_type = vcat(fill("natural", 1250), fill("nonnatural", 5000), fill("natural", 1250), fill("nonnatural", 5000)),
    learning_type = vcat(fill("social", 6250), fill("nonsocial", 6250))
)

model = lm(@formula(outcome ~ concept_type * learning_type), df_aov)

n_cond = vcat(fill(1, 1250), fill(2, 5000))
fig_bp_ob = Figure(fontsize=15, size = (588, 396));
axbp_ob = Axis(fig_bp_ob[1,1]; ylabel="AULC", xlabelsize=16, ylabelsize=16, xticks=(1:2, ["Natural", "Nonnatural"]), title="Output-based social updating", titlesize=22)
boxplot!(axbp_ob, n_cond, dropdims(vcat(nat_aulc_ob, non_nat_aulc_ob), dims=2), whiskerwidth = .5, width = 0.25, gap=-1.5, show_notch = false, show_outliers=true, color=(:lightsteelblue4, .5))
fig_bp_ob

# combined state-based and output-based averaging

@everywhere function updating_for_nets_combined(nets::Vector{Chain{Tuple{Dense{typeof(relu), Matrix{Float32}, Vector{Float32}}, Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}}}},
                                                ϵ₁::Float64,
                                                α₁::Float64,
                                                ϵ₂::Float64,
                                                α₂::Float64,
                                                labs::Vector{Int64})

    # find state peers and store average parameters (no social updating yet)
    peer_dist = params_sim(nets)
    p_mat = peer_dist .≥ 1 - ϵ₁
    p_means = Vector{VecOrMat{Float32}}[]
    for n in 1:numb_nets
        p = findall(==(1), p_mat[n, :])
        v = VecOrMat{Float32}[]
        for i in 1:4
            m = mean([ Flux.trainables(nets[j])[i] for j in p ], dims=1)[1]
            push!(v, m)
        end
        push!(p_means, v)
    end

    # worldly part of updating
    data = [ make_data(labs) for _ in 1:numb_nets ]
    for i in 1:numb_nets
        f, re = Flux.destructure(nets[i])
        opt = Flux.setup(Adam(), f)
        Flux.train!((p, x, y) -> Flux.logitcrossentropy(re(p)(x), y), f, data[i], opt)
        nets[i] = re(f)
    end

    # hk updating
    for n in 1:numb_nets
        for i in 1:4
            m = α₁ .* Flux.trainables(nets[n])[i] .+ (1. - α₁) .* p_means[n][i]
            Flux.trainables(nets[n])[i] .= m
        end
    end

    # find output peers and take modal responses
    best_guess = [ Flux.onecold(model(df_xs), 1:10) for model ∈ nets ]
    guess_sim = [ mutualinfo(best_guess[i], best_guess[j]) for i in 1:numb_nets, j in 1:numb_nets ]
    g_mat = guess_sim .≥ 1 - ϵ₂
    g_mat[diagind(g_mat)] .= 0
    modes = Vector{Int64}[]
    for j in 1:numb_nets
        bg = Vector{Int64}[]
        push!(bg, best_guess[j])
        bg_peers = best_guess[g_mat[j, :]]
        append!(bg, bg_peers)
        l = ceil(Int, α₂ * length(bg_peers))
        append!(bg, fill(best_guess[j], l))
        v = [ StatsBase.mode(getindex.(bg, i)) for i in 1:320 ]
        push!(modes, v)
    end

    return modes
end

@everywhere function combined_natural(; ϵ₁::Float64=.9, α₁::Float64=.81, ϵ₂::Float64=.07, α₂::Float64=.14, numb_epochs::Int=100)
    nets = [ Flux.Chain(Dense(3, 9, Flux.relu), Dense(9, 10)) for _ in 1:numb_nets ]
    res = [ updating_for_nets_combined(nets, ϵ₁, α₁, ϵ₂, α₂, ca.cluster) for _ in 1:numb_epochs ]
    return res
end

@everywhere function combined_non_natural(labs::Vector{Int64}; ϵ₁::Float64=.9, α₁::Float64=.81, ϵ₂::Float64=.07, α₂::Float64=.14, numb_epochs::Int=100)
    nets = [ Flux.Chain(Dense(3, 9, Flux.relu), Dense(9, 10)) for _ in 1:numb_nets ]
    res = [ updating_for_nets_combined(nets, ϵ₁, α₁, ϵ₂, α₂, labs) for _ in 1:numb_epochs ]
    return res
end

sim_nat_comb = pmap(_->combined_natural(), 1:25)
sim_non_nat_comb = pmap(i->combined_non_natural(random_clustering[i]), 1:100)

mn_comb = [ mean([ mutualinfo(ca.cluster, sim_nat_comb[k][j][i]) for i in 1:numb_nets ]) for j in 1:100, k in 1:25 ]
mnn_comb = [ mean([ mutualinfo(random_clustering[k], sim_non_nat_comb[k][j][i]) for i in 1:numb_nets ]) for j in 1:100, k in 1:100 ]

combined_bs_nat = [ Bootstrap.confint(bs(mn_comb[i, :]), BasicConfInt(.95))[1] for i in axes(mn_comb, 1) ]
combined_bs_non_nat = [ Bootstrap.confint(bs(mnn_comb[i, :]), BasicConfInt(.95))[1] for i in axes(mnn_comb, 1) ]

pvals_comb = [ pvalue(EqualVarianceTTest(mn_comb[i, :], mnn_comb[i, :])) for i in axes(mn_comb, 1) ]

cohend_comb = [ effectsize(CohenD(mn_comb[i, :], mnn_comb[i, :])) for i in axes(mn_comb, 1) ]

fco = Figure(fontsize=15, size=(780, 510));
ax1co = Axis(fco[1, 1], xlabel="Epoch", xlabelsize=16, ylabel="Mean NMI", ylabelsize=16, title="Combined social updating", titlesize=22)
ax2co = Axis(fco[1, 1], ylabel=rich("d"; font=:italic), ylabelsize=16, yaxisposition=:right)
l1co = lines!(ax1co, 1:size(mn_comb, 1), first.(combined_bs_nat), color=:lightsteelblue4)
b1co = band!(ax1co, 1:size(mn_comb, 1), getindex.(combined_bs_nat, 2), last.(combined_bs_nat), color=(:lightsteelblue4, .5))
l2co = lines!(ax1co, 1:size(mnn_comb, 1), first.(combined_bs_non_nat), color=:goldenrod)
b2co = band!(ax1co, 1:size(mnn_comb, 1), getindex.(combined_bs_non_nat, 2), last.(combined_bs_non_nat), color=(:goldenrod, .5))
l3co = scatter!(ax2co, 1:size(mnn_comb, 1), cohend_comb, markersize=7, color=[:indianred3, :olivedrab4][Int.(pvals .< .01) .+ 1])
Legend(fco[2, 1], [[l1co, b1co], [l2co, b2co], elem_1, elem_2], ["Natural", "Nonnatural", rich("Cohen's ", rich("d"; font=:italic), " (", rich("p"; font=:italic), " < .01)"), rich("Cohen's ", rich("d"; font=:italic), " (", rich("p"; font=:italic), " ≥ .01)")], framevisible=false, orientation = :horizontal, tellwidth = false, tellheight = true)
fco

nat_aulc_comb = reduce(vcat, [ mapslices(compute_aulc, [ mutualinfo(ca.cluster, sim_nat_comb[k][j][i]) for i in 1:numb_nets, j in 1:100 ], dims=2) for k in 1:25 ])
non_nat_aulc_comb = reduce(vcat, [ mapslices(compute_aulc, [ mutualinfo(random_clustering[k], sim_non_nat_comb[k][j][i]) for i in 1:numb_nets, j in 1:100 ], dims=2) for k in 1:100 ])

n_cond = vcat(fill(1, 1250), fill(2, 5000))
fig_bp_comb = Figure(fontsize=15, size = (588, 396));
axbp_comb = Axis(fig_bp_comb[1,1]; ylabel="AULC", xlabelsize=16, ylabelsize=16, xticks=(1:2, ["Natural", "Nonnatural"]), title="Combined social updating", titlesize=22)
boxplot!(axbp_comb, n_cond, dropdims(vcat(nat_aulc_comb, non_nat_aulc_comb), dims=2), whiskerwidth = .5, width = 0.25, gap=-1.5, show_notch = false, show_outliers=true, color=(:lightsteelblue4, .5))
fig_bp_comb