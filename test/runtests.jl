using Test
using SimpleLinear

using LinearAlgebra
using Random
using SparseArrays


@testset "Shift" begin
    m = [1 2; 3 4]
    v = [10, 100]
    @test m * v == [10 + 200, 30 + 400]

    @testset begin "constructor"
        m_shift = Shift(20, m)
        @test eltype(m_shift) == Int
        @test size(m_shift) == (2,2)
        @test size(m_shift, 1) == 2
        @test size(m_shift, 2) == 2
        @test m_shift * v == m * v + 20 .* v
        let y = [0, 0]
            mul!(y, m, v)
            @test y == [10 + 200, 30 + 400]
        end
        let y = [0, 0]
            mul!(y, m_shift, v)
            @test y == m * v + 20 .* v
        end

        let v = [1.0, sqrt(2.0)]
            y1 = [1983, 5]
            @test_throws InexactError mul!(y1, m_shift, v)
            y2 = [1983, 5.0]
            y3 = mul!(y2, m_shift, v)
            @test y3 === y2
            @test y2 == m * v + 20 .* v
        end


        m_shift = Shift{Int}(20, m)
        @test eltype(m_shift) == Int
        @test size(m_shift) == (2,2)
        @test size(m_shift, 1) == 2
        @test size(m_shift, 2) == 2
        @test m_shift * v == m * v + 20 .* v
        let y = [0, 0]
            mul!(y, m, v)
            @test y == [10 + 200, 30 + 400]
        end
        let y = [0, 0]
            mul!(y, m_shift, v)
            @test y == m * v + 20 .* v
        end

        m_shift = Shift{Float64}(20, m)
        @test eltype(m_shift) == Float64
        @test size(m_shift) == (2,2)
        @test size(m_shift, 1) == 2
        @test size(m_shift, 2) == 2
        @test m_shift * v == m * v + 20 .* v

        let y = [0, 0]
            mul!(y, m, v)
            @test y == [10 + 200, 30 + 400]
        end
        let y = [0, 0]
            @test mul!(y, m_shift, v) === y
            @test y == m * v + 20 .* v
        end
    end

    m3 = Shift(π, m)
    @test eltype(m3) == Float64
    @test m3 * v == m * v + π .* v
    let y1 = [1983, 5]
        @test_throws InexactError mul!(y1, m3, v)
        y2 = [1983, 5.0]
        y3 = mul!(y2, m3, v)
        @test y3 === y2
        @test y2 == m * v + π .* v
    end
    let v = [1.0, 1.3]
        y1 = [1983, 5]
        @test_throws InexactError mul!(y1, m3, v)
        y2 = [1983, 5.0]
        y3 = mul!(y2, m3, v)
        @test y3 === y2
        @test y2 == m * v + π .* v
    end

    m4 = Shift(4im, m)
    @test eltype(m4) == Complex{Int}
    @test m4 * v == m * v + 4im .* v
    let y1 = [1983, 5]
        @test_throws InexactError mul!(y1, m4, v)
        y2 = [1983, 5im]
        y3 = mul!(y2, m4, v)
        @test y3 === y2
        @test y2 == m * v + 4im .* v
    end

    m5 = Shift(π*im, m)
    @test eltype(m5) == ComplexF64
    @test m5 * v == m * v + π*im .* v
    let y1 = [1983, 5]
        @test_throws InexactError mul!(y1, m5, v)
        y1 = [1983, 5.0]
        @test_throws InexactError mul!(y1, m5, v)
        y1 = [1983, 5im]
        @test_throws InexactError mul!(y1, m5, v)
        y2 = [1983, π*im]
        y3 = mul!(y2, m5, v)
        @test y3 === y2
        @test y2 == m * v + π*im .* v
    end

    mp = [1 2π; 0 4]
    m6 = Shift(1im, mp)
    @test eltype(m6) == ComplexF64
    @test m6 * v == mp * v + im .* v
    let y1 = [1983, 5]
        @test_throws InexactError mul!(y1, m6, v)
        y1 = [1983, 5.0]
        @test_throws InexactError mul!(y1, m6, v)
        y1 = [1983, 5im]
        @test_throws InexactError mul!(y1, m6, v)
        y2 = [1983, π*im]
        y3 = mul!(y2, m6, v)
        @test y3 === y2
        @test y2 == mp * v + im .* v
    end

    @testset begin "Nested shift"
        m2 = Shift(20, Shift(-20, m))
        @test eltype(m2) == Int
        @test m2 * v == m * v
        let y1 = [1983, 5],
            y2 = [1983, 5]

            mul!(y1, m, v)
            mul!(y2, m2, v)
            @test y1 == y2
        end
    end

end

@testset "Invert" begin
    @testset "inverse type" begin
        m = [1 0; 2 3]
        f = lu(m)
        @test eltype(f) == Float64
        fi = LinearOperator(Float64, size(m), wrap_factorizedinvert!(f))
        @test eltype(fi) == Float64

        m = [1 0; 2 3im]
        f = lu(m)
        @test eltype(f) == ComplexF64
        fi = LinearOperator(ComplexF64, size(m), wrap_factorizedinvert!(f))
        @test eltype(fi) == ComplexF64
    end

    @testset "dense real" begin
        n = 16
        rng = MersenneTwister(0)
        x = rand(rng, Float64, (n, n))
        x = x + x'
        @test ishermitian(x)
        x += 10*I
        z = rand(rng, Float64, n)
        for FACTORIZE in [lu, qr, cholesky]
            xf = FACTORIZE(x)
            xfi = LinearOperator(Float64, (n,n), wrap_factorizedinvert!(xf))
            w = xfi * z
            @test x * w ≈ z
            fill!(w, NaN)
            @test mul!(w, xfi, z) === w
            @test x * w ≈ z
        end
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
            xfi = LinearOperator(Float64, (n,n), wrap_factorizedinvert!(xf))

            @test size(xfi) == size(x)
            @test size(xfi, 1) == size(x, 1)
            @test size(xfi, 2) == size(x, 2)
            w = xfi * z
            @test x * w ≈ z
            fill!(w, NaN)
            @test mul!(w, xfi, z) === w
            @test x * w ≈ z
        end
    end
end

@testset "iterative inverse-minres" begin
    # @testset "inverse type" begin
    #     @test eltype(IterativeInvertGMRes([1 2; 2 3])) == Float64
    #     @test eltype(IterativeInvertGMRes([1.0 2.0; 2.0 3.0])) == Float64
    #     @test eltype(IterativeInvertGMRes([1 2; 2+im 3])) == ComplexF64
    #     @test eltype(IterativeInvertGMRes([1.0 2.0; 2.0+1.0im 3.0])) == ComplexF64
    #     @test eltype(IterativeInvertMinRes([1 2; 2 3])) == Float64
    #     @test eltype(IterativeInvertMinRes([1.0 2.0; 2.0 3.0])) == Float64
    #     @test eltype(IterativeInvertMinRes([1 2-im; 2+im 3])) == ComplexF64
    #     @test eltype(IterativeInvertMinRes([1.0 2.0-1.0im; 2.0+1.0im 3.0])) == ComplexF64
    # end

    m = [-2.0 2.0+im; 2.0-im 2.0] # symmetric
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    v3 = [0.2, 0.8]
    inv_m0 = [-2/9  (2+im)/9; (2-im)/9 2/9]

    @testset "inverse test" begin
        # for INV in [IterativeInvertMinRes, IterativeInvertGMRes, x->iterativeinvert(x, :minres), x->iterativeinvert(x, :gmres)]
        #     inv_m = INV(m)
        inv_m = LinearOperator(ComplexF64, (2, 2), wrap_minres!(m))
        @test size(inv_m) == (2,2)
        @test size(inv_m, 1) == 2
        @test size(inv_m, 2) == 2
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
    end


    # eigenvalues = [-2.449489742783178, 2.449489742783178]

    # 0.750533+0.375266im  0.486519+0.243259im
    # -0.543945+0.0im       0.839121+0.0im

end
