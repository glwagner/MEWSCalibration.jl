using GLMakie

function compare_best_particle(eki, obs_with_kinetic_energy; resolution=(2400, 1200))
    #obs = eki.inverse_problem.observations
    times = obs_with_kinetic_energy.times
    Nt = length(times)
    b_obs = Array(interior(obs_with_kinetic_energy.field_time_serieses.b[Nt], 1, :, :))
    K_obs = Array(interior(obs_with_kinetic_energy.field_time_serieses.K[Nt], 1, :, :))

    xo, yo, zo = nodes(obs_with_kinetic_energy.field_time_serieses.b)

    @show err, i = findmin(eki.iteration_summaries[end].mean_square_errors)
    b_mod = Array(interior(eki.inverse_problem.simulation.model.tracers.b, i, :, :))
    K_mod = Array(interior(eki.inverse_problem.simulation.model.tracers.K, i, :, :))

    xm, ym, zm = nodes(eki.inverse_problem.simulation.model.tracers.b)

    @show Klim = maximum(K_mod)

    # Plot
    fig = Figure(; resolution)
    ax_obs = Axis(fig[1, 1], title="Observations")
    heatmap!(ax_obs, yo, zo, K_obs, colormap=:solar, colorrange=(0, Klim))
    contour!(ax_obs, yo, zo, b_obs, levels=25, color=:black, linewidth=2)

    ax_mod = Axis(fig[2, 1], title="MEWS")
    heatmap!(ax_mod, ym, zm, K_mod, colormap=:solar, colorrange=(0, Klim))
    contour!(ax_mod, ym, zm, b_mod, levels=25, color=:black, linewidth=2)

    return fig
end

fig = compare_best_particle(eki, obs_with_kinetic_energy)
display(fig)
