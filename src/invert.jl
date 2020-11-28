export IterativeInvertMinRes
export IterativeInvertGMRes
export iterativeinvert

export FactorizeInvert
export factorizeinvert

import LinearAlgebra
import SuiteSparse
import IterativeSolvers

abstract type AbstractInvert{S}<:AbstractMatrix{S} end

FloatOrComplexFloat = Union{<:AbstractFloat, <:Complex{<:AbstractFloat}}
IntegerOrComplexInteger = Union{<:Integer, <:Complex{<:Integer}}

invtype(::Type{T}) where {T<:Integer} = Float64
invtype(::Type{T}) where {T<:AbstractFloat} = T
invtype(::Type{Complex{T}}) where {T<:Integer} = ComplexF64
invtype(::Type{Complex{T}}) where {T<:AbstractFloat} = Complex{T}


struct FactorizeInvert{S<:FloatOrComplexFloat, F<:LinearAlgebra.Factorization{S}}<:AbstractInvert{S}
    factorization::F
    function FactorizeInvert(factorization::F) where {S<:FloatOrComplexFloat, F<:LinearAlgebra.Factorization{S}}
        return new{S, F}(factorization)
    end
end

function Base.:(*)(m::FactorizeInvert, v::AbstractVector)
    U = promote_type(eltype(m), eltype(v))
    w = zeros(U, length(v))
    return LinearAlgebra.ldiv!(w, m.factorization, v)
end

function LinearAlgebra.mul!(y::AbstractVector{S}, m::FactorizeInvert, x::AbstractVector) where {S}
    fill!(y, zero(S))
    return LinearAlgebra.ldiv!(y, m.factorization, x)
end

Base.size(s::FactorizeInvert, args...) = size(s.factorization, args...)

# workaround for missing ldiv!
function LinearAlgebra.ldiv!(A::AbstractVector, F::SuiteSparse.SPQR.QRSparse, B::AbstractVector)
    A[:] .= F \ B
    return A
end


"""
    factorizeinvert(m; algorithm=:lu, kwargs...)
"""
function factorizeinvert(m::AbstractMatrix; algorithm::Symbol=:lu, kwargs...)
    if algorithm == :lu
        return FactorizeInvert(LinearAlgebra.lu(m; kwargs...))
    elseif algorithm == :qr
        return FactorizeInvert(LinearAlgebra.qr(m; kwargs...))
    elseif algorithm == :cholesky
        return FactorizeInvert(LinearAlgebra.cholesky(m; kwargs...))
    elseif algorithm == :ldlt
        return FactorizeInvert(LinearAlgebra.ldlt(m; kwargs...))
    else
        throw(ArgumentError("Unsupported algorithm $algorithm"))
    end
end


struct IterativeInvertMinRes{
    S<:FloatOrComplexFloat,
    L<:AbstractMatrix
}<:AbstractInvert{S}

    operation::L
    kwargs::Vector{Pair{Symbol, Any}}
    # skew_hermitian::Bool
    # tol::Real
    # maxiter::Int
    # Pl
    # Pr
    # log::Bool
    # verbose::Bool

    function IterativeInvertMinRes(operation::L; kwargs...) where {L<:AbstractMatrix}
        # !LinearAlgebra.ishermitian(operation) && throw(ArgumentError("operation must be hermitian"))
        S = invtype(valtype(operation))
        if isempty(kwargs)
            return new{S, L}(operation, Pair{Symbol, Any}[])
        else
            return new{S, L}(operation, kwargs)
        end
    end

    function IterativeInvertMinRes{S}(operation::L; kwargs...) where {S, L<:AbstractMatrix}
        # !LinearAlgebra.ishermitian(operation) && throw(ArgumentError("operation must be hermitian"))
        if isempty(kwargs)
            return new{S, L}(operation, Pair{Symbol, Any}[])
        else
            return new{S, L}(operation, kwargs)
        end
    end

    # function Invert(operation::AbstractMatrix{S};
    #     skew_hermitian::Bool=false,
    #     tol::Real=Base.rtoldefault(real(S)),
    #     maxiter::Int=size(operation, 2),
    #     Pl=IterativeSolvers.Identity(),
    #     Pr=IterativeSolvers.Identity(),
    #     log::Bool=false,
    #     verbose::Bool=false
    # ) where {S}
end

# function Base.:(*)(m::IterativeInvertMinRes, v::AbstractVector{<:IntegerOrComplexInteger})
# 	return IterativeSolvers.minres(m.operation, float.(v); m.kwargs...)
# end

function Base.:(*)(m::IterativeInvertMinRes, v::AbstractVector{<:FloatOrComplexFloat})
	return IterativeSolvers.minres(m.operation, v; m.kwargs...)
end


# function LinearAlgebra.mul!(y::AbstractVector{S}, m::IterativeInvertMinRes, x::AbstractVector{<:IntegerOrComplexInteger}) where {S}
#     fill!(y, zero(S))
# 	IterativeSolvers.minres!(y, m.operation, float.(x); initially_zero=true, m.kwargs...)
# end

function LinearAlgebra.mul!(y::AbstractVector{S}, m::IterativeInvertMinRes, x::AbstractVector{<:FloatOrComplexFloat}) where {S}
    fill!(y, zero(S))
    result, history = IterativeSolvers.minres!(y, m.operation, x; initially_zero=true, log=true, m.kwargs...)
    history.isconverged || @warn "minres! not converged: $history"
    return result
end

Base.size(s::IterativeInvertMinRes, args...) = size(s.operation, args...)

struct IterativeInvertGMRes{
    S<:FloatOrComplexFloat,
    L<:AbstractMatrix
}<:AbstractMatrix{S}
    operation::L
    kwargs::Vector{Pair{Symbol, Any}}

    function IterativeInvertGMRes(operation::L; kwargs...) where {L<:AbstractMatrix}
        S = invtype(eltype(operation))
        if isempty(kwargs)
            return new{S, L}(operation, Pair{Symbol, Any}[])
        else
            return new{S, L}(operation, kwargs)
        end
    end
end

function Base.:(*)(m::IterativeInvertGMRes, v::AbstractVector{<:AbstractFloat})
    return IterativeSolvers.gmres(m.operation, v; m.kwargs...)
end

# function Base.:(*)(m::IterativeInvertGMRes, v::AbstractVector{<:Integer})
#     return IterativeSolvers.gmres(m.operation, float.(v); m.kwargs...)
# end

function LinearAlgebra.mul!(y::AbstractVector{S}, m::IterativeInvertGMRes, x::AbstractVector{<:FloatOrComplexFloat}) where {S}
    fill!(y, zero(S))
    IterativeSolvers.gmres!(y, m.operation, x; initially_zero=true, m.kwargs...)
    return y
end

function iterativeinvert(operation::L, algorithm::Symbol=:minres; kwargs...) where {L<:AbstractMatrix}
    S = eltype(operation)
    if algorithm == :minres
        Alg = IterativeInvertMinRes
    elseif algorithm == :gmres
        Alg = IterativeInvertGMRes
    else
        Alg = nothing
        throw(ArgumentError("unsupported algorithm $algorithm"))
    end
    return Alg(operation; kwargs...)
end
