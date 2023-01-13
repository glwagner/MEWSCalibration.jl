using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: SliceEnsembleSize
using Oceananigans.TurbulenceClosures.MEWSVerticalDiffusivities: MEWSVerticalDiffusivity
using ParameterEstimocean
using ParameterEstimocean.PseudoSteppingSchemes: Kovachki2018InitialConvergenceRatio

filename = "simple_baroclinic_adjustment_dy100km_zonal_average.jld2"

artifacts_url = "https://github.com/glwagner/MEWSCalibrationArtifacts/raw/main/baroclinic_adjustment/"
filename_url = joinpath(artifacts_url, filename)
isfile(filename) || Base.download(filename_url, filename)

arch = CPU()
Nens = 20
Ly = 2000kilometers
Lz = 1kilometer
Ny = 256
Nz = 32
Cᴰ = 2e-3

regrid = RectilinearGrid(arch,
                         size = (64, 32),
                         topology = (Flat, Bounded, Bounded),
                         y = (-Ly/2, Ly/2),
                         z = (-Lz, 0),
                         halo = (3, 3))

space_transformation = SpaceIndices(x=:, y=1:4:Ny, z=:)
#transformation = Transformation(; space=space_transformation)
transformation = Transformation()
field_names = tuple(:b) # (:b, :K)
times = [10days, 20days, 30days]
observations = SyntheticObservations(filename; times, transformation, field_names, regrid)
obs_with_kinetic_energy = SyntheticObservations(filepath; times, transformation, field_names=(:b, :k), regrid)

@show observations

coriolis = BetaPlane(; observations.metadata.coriolis...)

slice_ensemble_size = SliceEnsembleSize(size=(regrid.Ny, regrid.Nz), ensemble=Nens)

ensemble_grid = RectilinearGrid(arch,
                                size = slice_ensemble_size,
                                topology = (Flat, Bounded, Bounded),
                                y = (-Ly/2, Ly/2),
                                z = (-Lz, 0),
                                halo = (3, 3))

@show ensemble_grid

mews = MEWSVerticalDiffusivity(; Cᴰ)
mews_ensemble = [deepcopy(mews) for ω = 1:Nens]

ensemble_model = HydrostaticFreeSurfaceModel(grid = ensemble_grid,
                                             tracers = (:b, :K),
                                             buoyancy = BuoyancyTracer(),
                                             coriolis = coriolis,
                                             closure = mews_ensemble,
                                             free_surface = ImplicitFreeSurface())

@show ensemble_model

simulation = Simulation(ensemble_model; Δt=20minutes, stop_time=times[end])

priors = (;
    Cʰ  = ScaledLogitNormal(bounds = (0.0, 20.0)),
    Cᴷʰ = ScaledLogitNormal(bounds = (0.0, 20.0)),
    Cᴷᶻ = ScaledLogitNormal(bounds = (0.0, 20.0)),
    Cⁿ  = ScaledLogitNormal(bounds = (0.0, 20.0)),
)

free_parameters = FreeParameters(priors)

function initialize_simulation(sim, parameters)
    K = sim.model.tracers.K
    Kᵢ = obs_with_kinetic_energy.field_time_serieses.k[1]
    interior(K, :, :, :) .= interior(Kᵢ, :, :, :)
    return nothing
end
    
calibration = InverseProblem(observations, simulation, free_parameters; initialize_simulation)

@show calibration

initial_convergence_ratio = 0.7
pseudo_stepping = Kovachki2018InitialConvergenceRatio(; initial_convergence_ratio)
resample_failure_fraction = 0.2
acceptable_failure_fraction = 1.0
resampler = Resampler(; resample_failure_fraction, acceptable_failure_fraction)
noise_covariance = 1.0

eki = EnsembleKalmanInversion(calibration; noise_covariance, resampler, pseudo_stepping)

@show eki

for n = 1:10
    iterate!(eki)
    @show eki.iteration_summaries[end]
end
