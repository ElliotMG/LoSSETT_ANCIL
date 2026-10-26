using LoSSETT
using DelimitedFiles

const SPHERE_RADIUS_METRES = 6_371_000.0
const LENGTH_SCALE_METRES = 500_000.0
const MEAN_MAP_LENGTH_SCALE_METRES = 250_000.0
const CARTESIAN_METRES_PER_DEGREE = 110_000.0
const CARTESIAN_MAX_RADIUS = 10.0 * CARTESIAN_METRES_PER_DEGREE
const SPHERICAL_MAX_RADIUS = deg2rad(10.0) * SPHERE_RADIUS_METRES
const DEFAULT_MAP_PRESSURE_HPA = 200.0

pressure_text(pressure_hpa) =
    isinteger(pressure_hpa) ? string(Int(pressure_hpa)) : string(pressure_hpa)
pressure_tag(pressure_hpa) = "$(pressure_text(pressure_hpa))hPa"

function read_manifest(path)
    manifest = Dict{String, String}()
    for line in eachline(path)
        isempty(line) && continue
        key, value = split(line, '='; limit=2)
        manifest[key] = value
    end
    get(manifest, "format", "") == "lossett-tutorial-v1" ||
        error("Unsupported tutorial input format in $path")
    return manifest
end

parse_vector(manifest, key, element_type) =
    parse.(element_type, split(manifest[key], ','))

function read_velocity(path, shape)
    bytes = read(path)
    expected_bytes = prod(shape) * sizeof(Float64)
    length(bytes) == expected_bytes ||
        error("Expected $expected_bytes bytes in $path; found $(length(bytes))")
    return reshape(collect(reinterpret(Float64, bytes)), shape)
end

function write_contour_csv(path, field)
    open(path, "w") do io
        for latitude_index in axes(field, 1)
            for longitude_index in axes(field, 2)
                longitude_index > 1 && print(io, ',')
                print(io, field[latitude_index, longitude_index])
            end
            println(io)
        end
    end
end

function calculate(u, v, w, longitude, latitude, geometry, length_scale=LENGTH_SCALE_METRES)
    if geometry == :cartesian
        return kinetic_energy_transfer(
            u,
            v,
            w,
            longitude .* CARTESIAN_METRES_PER_DEGREE,
            latitude .* CARTESIAN_METRES_PER_DEGREE,
            [length_scale];
            max_radius=10.0 * CARTESIAN_METRES_PER_DEGREE,
            periodic=(true, false),
            xdim=4,
            ydim=3,
        )
    elseif geometry == :tangent_quadratic
        return kinetic_energy_transfer(
            u,
            v,
            w,
            longitude,
            latitude,
            [length_scale];
            max_radius=SPHERICAL_MAX_RADIUS,
            geometry=:tangent_quadratic,
            sphere_radius=SPHERE_RADIUS_METRES,
            xdim=4,
            ydim=3,
            use_angular_weights=false,
        )
    end
    error("Unknown geometry: $geometry")
end

function geometry_label(geometry)
    geometry == :tangent_quadratic && return "Tangent quadratic"
    return uppercasefirst(string(geometry))
end

function calculate_and_save(input_directory, u, v, w,
                            longitude, latitude, pressure_index,
                            geometry, map_pressure_hpa;
                            length_scale=LENGTH_SCALE_METRES,
                            filename_geometry=geometry)
    core_result() = calculate(u, v, w, longitude, latitude, geometry, length_scale)

    warmup = core_result()
    warmup_sum = sum(warmup.transfer)
    isfinite(warmup_sum) || error("Julia $geometry warm-up produced a non-finite transfer")
    started_ns = time_ns()
    result = core_result()
    materialized_sum = sum(result.transfer)
    isfinite(materialized_sum) || error("Julia $geometry result is non-finite")
    elapsed_seconds = (time_ns() - started_ns) / 1e9

    scale_index = findfirst(
        scale -> isapprox(scale, length_scale; atol=1e-8, rtol=0),
        result.length_scales,
    )
    scale_index === nothing &&
        error("Julia $geometry core did not return the requested scale")
    field = result.transfer[scale_index, 1, pressure_index, :, :]

    tag = pressure_tag(map_pressure_hpa)
    result_path = if isapprox(length_scale, LENGTH_SCALE_METRES; atol=1e-8, rtol=0)
        joinpath(input_directory, "$(filename_geometry)_julia_$(tag).csv")
    else
        scale_tag = round(Int, length_scale / 1000)
        joinpath(input_directory, "$(filename_geometry)_julia_$(scale_tag)km.csv")
    end
    write_contour_csv(result_path, field)
    println(
        "LoSSETT.jl $(geometry_label(geometry)) core runtime: ",
        round(elapsed_seconds; digits=3),
        " s",
    )
    return result_path, elapsed_seconds
end

function plot_python_result(python, plotter, result_path, manifest_path,
                            label, output_path, elapsed_seconds)
    run(Cmd([
        python,
        plotter,
        "--plot-result",
        result_path,
        "--manifest",
        manifest_path,
        "--runtime",
        string(elapsed_seconds),
        "--label",
        label,
        "--output",
        output_path,
    ]))
end

function plot_difference(python, plotter, manifest_path, input_directory,
                         geometry, map_pressure_hpa)
    tag = pressure_tag(map_pressure_hpa)
    run(Cmd([
        python,
        plotter,
        "--plot-difference",
        string(geometry),
        "--manifest",
        manifest_path,
        "--python-result",
        joinpath(input_directory, "$(geometry)_python_$(tag).csv"),
        "--julia-result",
        joinpath(input_directory, "$(geometry)_julia_$(tag).csv"),
    ]))
end

function plot_comparison(python, plotter, manifest_path, input_directory,
                         output_path, map_pressure_hpa)
    tag = pressure_tag(map_pressure_hpa)
    run(Cmd([
        python,
        plotter,
        "--plot-comparison",
        "--manifest",
        manifest_path,
        "--comparison-output",
        output_path,
        "--cartesian-python-result",
        joinpath(input_directory, "cartesian_python_$(tag).csv"),
        "--cartesian-julia-result",
        joinpath(input_directory, "cartesian_julia_$(tag).csv"),
        "--tangent-python-result",
        joinpath(input_directory, "tangent_quadratic_python_$(tag).csv"),
        "--tangent-julia-result",
        joinpath(input_directory, "tangent_quadratic_julia_$(tag).csv"),
    ]))
end

function process_bundle(input_directory, python, plotter, timing_io)
    manifest_path = joinpath(input_directory, "manifest.txt")
    manifest = read_manifest(manifest_path)
    shape = Tuple(parse.(Int, split(manifest["shape"], ',')))
    length(shape) == 4 ||
        error("Expected input dimensions (time, pressure, latitude, longitude)")
    Base.ENDIAN_BOM == 0x04030201 ||
        error("The shared input bundle requires a little-endian Julia runtime")

    pressure = parse_vector(manifest, "pressure", Float64)
    latitude = parse_vector(manifest, "latitude", Float64)
    longitude = parse_vector(manifest, "longitude", Float64)
    map_pressure_hpa = parse(Float64, manifest["map_pressure_hpa"])
    any(abs.(latitude) .>= 90.0) &&
        error("Tangent-quadratic tutorial geometry does not support polar grid points")
    parse(Float64, manifest["length_scale_m"]) == LENGTH_SCALE_METRES ||
        error("Input manifest must specify the 500 km map scale")
    length(pressure) == 1 && isapprox(pressure[1], map_pressure_hpa; atol=0.01, rtol=0) ||
        error("Input manifest pressure coordinate must match its requested map level")
    shape == (1, length(pressure), length(latitude), length(longitude)) ||
        error("Input manifest coordinates do not match its declared shape")

    u = read_velocity(joinpath(input_directory, "u.f64le"), shape)
    v = read_velocity(joinpath(input_directory, "v.f64le"), shape)
    w = read_velocity(joinpath(input_directory, "w.f64le"), shape)
    pressure_index = findfirst(
        level -> isapprox(level, map_pressure_hpa; atol=0.01, rtol=0),
        pressure,
    )
    pressure_index === nothing &&
        error("Pressure level $(map_pressure_hpa) hPa is missing from the input bundle")

    cartesian_path, cartesian_runtime = calculate_and_save(
        input_directory, u, v, w, longitude, latitude,
        pressure_index, :cartesian, map_pressure_hpa,
    )
    tangent_quadratic_path, tangent_quadratic_runtime = calculate_and_save(
        input_directory, u, v, w, longitude, latitude,
        pressure_index, :tangent_quadratic, map_pressure_hpa,
    )
    date_compact = replace(manifest["date"], "-" => "")
    tag = pressure_tag(map_pressure_hpa)

    python = get(ENV, "PYTHON", "python")
    plotter = joinpath(@__DIR__, "Tutorial.py")
    plot_python_result(
        python, plotter, cartesian_path, manifest_path,
        "LoSSETT.jl Cartesian ($(pressure_text(map_pressure_hpa)) hPa, 500 km)",
        joinpath(input_directory, "Tutorial_cartesian_julia_$(date_compact)_$(tag).png"),
        cartesian_runtime,
    )
    plot_python_result(
        python, plotter, tangent_quadratic_path, manifest_path,
        "LoSSETT.jl Tangent quadratic ($(pressure_text(map_pressure_hpa)) hPa, 500 km)",
        joinpath(input_directory, "Tutorial_tangent_quadratic_julia_$(date_compact)_$(tag).png"),
        tangent_quadratic_runtime,
    )
    plot_difference(
        python, plotter, manifest_path, input_directory, :cartesian, map_pressure_hpa
    )
    plot_difference(
        python, plotter, manifest_path, input_directory, :tangent_quadratic,
        map_pressure_hpa,
    )
    plot_comparison(
        python,
        plotter,
        manifest_path,
        input_directory,
        joinpath(input_directory, "Tutorial_comparison_$(date_compact)_$(tag).png"),
        map_pressure_hpa,
    )
    mean_path, mean_runtime = calculate_and_save(
        input_directory, u, v, w, longitude, latitude,
        pressure_index, :cartesian, map_pressure_hpa;
        length_scale=MEAN_MAP_LENGTH_SCALE_METRES,
        filename_geometry="Dl_u",
    )
    println(timing_io, "$(manifest[\"selected_time\"]),julia,cartesian,$(cartesian_runtime)")
    println(timing_io, "$(manifest[\"selected_time\"]),julia,tangent_quadratic,$(tangent_quadratic_runtime)")
    println(timing_io, "$(manifest[\"selected_time\"]),julia,cartesian_250km_mean_input,$(mean_runtime)")
    return mean_path, latitude, longitude
end

function main(args)
    length(args) <= 1 ||
        error("Usage: julia Tutorial.jl [monthly-output-directory]")
    default_directory = "Tutorial_input_20050101_$(pressure_tag(DEFAULT_MAP_PRESSURE_HPA))"
    root = abspath(isempty(args) ? default_directory : args[1])
    directories = if isfile(joinpath(root, "manifest.txt"))
        [root]
    else
        sort(filter(isdir, readdir(root; join=true)))
    end
    isempty(directories) && error("No tutorial input bundles found in $root")
    python = get(ENV, "PYTHON", "python")
    plotter = joinpath(@__DIR__, "Tutorial.py")
    timing_path = joinpath(root, "julia_timings.csv")
    open(timing_path, "w") do timing_io
        println(timing_io, "selected_time,implementation,geometry,seconds")
        mean_fields = Matrix{Float64}[]
        mean_latitude = Float64[]
        mean_longitude = Float64[]
        for (index, input_directory) in enumerate(directories)
            isfile(joinpath(input_directory, "manifest.txt")) || continue
            println("Processing Julia bundle $(index)/$(length(directories)): $input_directory")
            mean_path, mean_latitude, mean_longitude =
                process_bundle(input_directory, python, plotter, timing_io)
            push!(mean_fields, Float64.(readdlm(mean_path, ',')))
        end
        isempty(mean_fields) && error("No Julia mean fields were produced")
        julia_mean = reduce(+, mean_fields) ./ length(mean_fields)
        mean_root = root
        mean_path = joinpath(mean_root, "Dl_u_julia_january_mean_250km.csv")
        write_contour_csv(mean_path, julia_mean)
        python_mean = joinpath(mean_root, "Dl_u_python_january_mean_250km.csv")
        mean_manifest = joinpath(first(directories), "manifest.txt")
        run(Cmd([
            python, plotter, "--plot-time-mean", "--manifest", mean_manifest,
            "--mean-python-result", python_mean, "--mean-julia-result", mean_path,
            "--mean-output", joinpath(mean_root, "Dl_u_january_mean_250km_comparison.png"),
        ]))
    end
    println("Julia timings written to $timing_path")
end

main(ARGS)
