using LoSSETT

const SPHERE_RADIUS_METRES = 6_371_000.0
const LENGTH_SCALE_METRES = 500_000.0
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

function calculate(u, v, w, longitude, latitude, geometry)
    if geometry == :cartesian
        return kinetic_energy_transfer(
            u,
            v,
            w,
            longitude .* CARTESIAN_METRES_PER_DEGREE,
            latitude .* CARTESIAN_METRES_PER_DEGREE,
            [LENGTH_SCALE_METRES];
            max_radius=CARTESIAN_MAX_RADIUS,
            periodic=(true, false),
            xdim=4,
            ydim=3,
        )
    elseif geometry == :spherical
        return kinetic_energy_transfer(
            u,
            v,
            w,
            longitude,
            latitude,
            [LENGTH_SCALE_METRES];
            max_radius=SPHERICAL_MAX_RADIUS,
            geometry=:spherical,
            sphere_radius=SPHERE_RADIUS_METRES,
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
            [LENGTH_SCALE_METRES];
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
                            geometry, map_pressure_hpa)
    core_result() = calculate(u, v, w, longitude, latitude, geometry)

    warmup = core_result()
    warmup_sum = sum(warmup.transfer)
    isfinite(warmup_sum) || error("Julia $geometry warm-up produced a non-finite transfer")
    started_ns = time_ns()
    result = core_result()
    materialized_sum = sum(result.transfer)
    isfinite(materialized_sum) || error("Julia $geometry result is non-finite")
    elapsed_seconds = (time_ns() - started_ns) / 1e9

    scale_index = findfirst(
        scale -> isapprox(scale, LENGTH_SCALE_METRES; atol=1e-8, rtol=0),
        result.length_scales,
    )
    scale_index === nothing &&
        error("Julia $geometry core did not return the requested 500 km scale")
    field = result.transfer[scale_index, 1, pressure_index, :, :]

    tag = pressure_tag(map_pressure_hpa)
    result_path = joinpath(input_directory, "$(geometry)_julia_$(tag).csv")
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

function main(args)
    length(args) <= 1 ||
        error("Usage: julia Tutorial.jl [input-directory] (default pressure: 200 hPa)")
    default_directory = "Tutorial_input_20160801_$(pressure_tag(DEFAULT_MAP_PRESSURE_HPA))"
    input_directory = abspath(isempty(args) ? default_directory : args[1])
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
    spherical_path, spherical_runtime = calculate_and_save(
        input_directory, u, v, w, longitude, latitude,
        pressure_index, :spherical, map_pressure_hpa,
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
        joinpath(pwd(), "Tutorial_cartesian_julia_$(date_compact)_$(tag).png"),
        cartesian_runtime,
    )
    plot_python_result(
        python, plotter, spherical_path, manifest_path,
        "LoSSETT.jl Spherical ($(pressure_text(map_pressure_hpa)) hPa, 500 km)",
        joinpath(pwd(), "Tutorial_spherical_julia_$(date_compact)_$(tag).png"),
        spherical_runtime,
    )
    plot_python_result(
        python, plotter, tangent_quadratic_path, manifest_path,
        "LoSSETT.jl Tangent quadratic ($(pressure_text(map_pressure_hpa)) hPa, 500 km)",
        joinpath(pwd(), "Tutorial_tangent_quadratic_julia_$(date_compact)_$(tag).png"),
        tangent_quadratic_runtime,
    )
    plot_difference(
        python, plotter, manifest_path, input_directory, :cartesian, map_pressure_hpa
    )
    plot_difference(
        python, plotter, manifest_path, input_directory, :spherical, map_pressure_hpa
    )
    plot_difference(
        python, plotter, manifest_path, input_directory, :tangent_quadratic,
        map_pressure_hpa,
    )
end

main(ARGS)
