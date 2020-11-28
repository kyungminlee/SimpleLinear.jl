using Test
using SimpleLinear

using LinearAlgebra
using Random
using SparseArrays

@testset "Shift" begin
    m = [1 2; 3 4]
    v = [10, 100]
    m_shift = Shift(20, m)
    @test m * v == [10 + 200, 30 + 400]
    @test m_shift * v == m * v + 20 .* v

    let y = [0, 0]
        mul!(y, m, v)
        @test y == [10 + 200, 30 + 400]
    end
    let y = [0, 0]
        mul!(y, m_shift, v)
        @test y == m * v + 20 .* v
    end

    m2 = Shift(20, Shift(-20, m))
    @test m2 * v == m * v
    let y1 = [1983, 5],
        y2 = [1983, 5]

        mul!(y1, m, v)
        mul!(y2, m2, v)
        @test y1 == y2
    end
end

@testset "FactorizeInvert" begin
    @testset "dense" begin
        n = 16
        rng = MersenneTwister(0)
        x = rand(rng, Float64, (n, n))
        x = x + x'
        @test ishermitian(x)
        xf = lu(x)
        xfi = FactorizeInvert(xf)
        z = rand(rng, Float64, n)
        w = xfi * z
        @test x * w ≈ z
        fill!(w, NaN)
        @test mul!(w, xfi, z) === w
        @test x * w ≈ z
    end
    @testset "sparse" begin
        n = 16
        m = 4
        rng = MersenneTwister(0)
        x = sparse(rand(rng, 1:n, m*n), rand(rng, 1:n, m*n), rand(rng, Float64, m*n))
        x = x + x'
        @test ishermitian(x)

        z = rand(rng, Float64, n)
        for FACTORIZE in [lu, qr]
            xf = FACTORIZE(x)
            xfi = FactorizeInvert(xf)
            w = xfi * z
            @test x * w ≈ z
            fill!(w, NaN)
            @test mul!(w, xfi, z) === w
            @test x * w ≈ z
        end
    end
end

@testset "IterativeInvertMinRes" begin
    m = [-2.0 2.0+im; 2.0-im 2.0] # symmetric
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    v3 = [0.2, 0.8]
    inv_m0 = [-2/9  (2+im)/9; (2-im)/9 2/9]
    inv_m = IterativeInvertMinRes(m)
    @test isapprox(inv_m * v1, inv_m0 * v1)
    @test isapprox(inv_m * v2, inv_m0 * v2)
    @test isapprox(inv_m * v3, inv_m0 * v3)

    let y1 = zeros(ComplexF64, 2)
        y2 = zeros(ComplexF64, 2)
        mul!(y1, inv_m0, v1)
        mul!(y2, inv_m , v1)
        @test isapprox(y1, y2)
    end
    let y1 = zeros(ComplexF64, 2)
        y2 = zeros(ComplexF64, 2)
        mul!(y1, inv_m0, v2)
        mul!(y2, inv_m , v2)
        @test isapprox(y1, y2)
    end
    let y1 = zeros(ComplexF64, 2)
        y2 = zeros(ComplexF64, 2)
        mul!(y1, inv_m0, v3)
        mul!(y2, inv_m , v3)
        @test isapprox(y1, y2)
    end




    # eigenvalues = [-2.449489742783178, 2.449489742783178]

    # 0.750533+0.375266im  0.486519+0.243259im
    # -0.543945+0.0im       0.839121+0.0im

end
