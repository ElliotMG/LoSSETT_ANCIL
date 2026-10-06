using LoSSETT

const SPHERE_RADIUS_METRES = 6_371_000.0
const LENGTH_SCALE_METRES = 500_000.0
const MAX_RADIUS_METRES = deg2rad(10.0) * SPHERE_RADIUS_METRES
const MAP_PRESSURE_HPA = 850.0

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

calculate(u, v, w, longitude, latitude) = kinetic_energy_transfer(
    u,
    v,
    w,
    longitude,
    latitude,
    [LENGTH_SCALE_METRES];
    max_radius=MAX_RADIUS_METRES,
    geometry=:spherical,
    sphere_radius=SPHERE_RADIUS_METRES,
    xdim=4,
    ydim=3,
)

function main(args)
    length(args) <= 1 || error("Usage: julia Tutorial.jl [input-directory]")
    input_directory = abspath(isempty(args) ? "Tutorial_input_20160801" : args[1])
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
    shape == (1, length(pressure), length(latitude), length(longitude)) ||
        error("Input manifest coordinates do not match its declared shape")

    u = read_velocity(joinpath(input_directory, "u.f64le"), shape)
    v = read_velocity(joinpath(input_directory, "v.f64le"), shape)
    w = read_velocity(joinpath(input_directory, "w.f64le"), shape)
    pressure_index = findfirst(==(MAP_PRESSURE_HPA), pressure)
    pressure_index === nothing &&
        error("Pressure level $(MAP_PRESSURE_HPA) hPa is missing from the input bundle")

    warmup_result = calculate(u, v, w, longitude, latitude)
    warmup_sum = sum(warmup_result.transfer)
    isinf(warmup_sum) && error("Julia warm-up produced an infinite transfer")
    started_ns = time_ns()
    result = calculate(u, v, w, longitude, latitude)
    materialized_sum = sum(result.transfer)
    isinf(materialized_sum) && error("Julia calculation produced an infinite transfer")
    elapsed_seconds = (time_ns() - started_ns) / 1e9

    scale_index = findfirst(
        scale -> isapprox(scale, LENGTH_SCALE_METRES; atol=1e-8, rtol=0),
        result.length_scales,
    )
    scale_index === nothing &&
        error("The Julia core did not return the requested 500 km scale")
    field = result.transfer[scale_index, 1, pressure_index, :, :]

    date_string = manifest["date"]
    date_compact = replace(date_string, "-" => "")
    result_path = joinpath(input_directory, "julia_transfer_$date_compact.csv")
    write_contour_csv(result_path, field)

    python = get(ENV, "PYTHON", "python")
    plotter = joinpath(@__DIR__, "Tutorial.py")
    run(Cmd([
        python,
        plotter,
        "--plot-result",
        result_path,
        "--manifest",
        manifest_path,
        "--runtime",
        string(elapsed_seconds),
        "--implementation",
        "LoSSETT.jl",
    ]))
    println("LoSSETT.jl core runtime: ", round(elapsed_seconds; digits=3), " s")
end

main(ARGS)
