import IterativeSolvers
import LinearAlgebra

export Invert
export InvertMinRes
export InvertGMRes

abstract type AbstractInvert{S, L<:AbstractMatrix{S}}<:AbstractMatrix{S} end

struct InvertMinRes{S, L<:AbstractMatrix{S}}<:AbstractInvert{S, L}
    operation::L
    kwargs::Vector{Pair{Symbol, Any}}

    function Invert(operation::L; kwargs...) where {L<:AbstractMatrix}
        S = valtype(operation)
        if isempty(kwargs)
            return new{S, L}(operation, Pair{Symbol, Any}[])
        else
            return new{S, L}(operation, kwargs)
        end
    end
end

function Base.:(*)(m::InvertMinRes, v::AbstractVector)
	return minres(m.operation, v; m.kwargs...)
end

function LinearAlgebra.mul!(y::AbstractVector{S}, m::InvertMinRes, x::AbstractVector) where {S}
    fill!(y, zero(S))
	minres!(y, m.operation, x; initially_zero=true, m.kwargs...)
end

struct InvertGMRes{S, L<:AbstractMatrix{S}}<:AbstractMatrix{S}
    operation::L
    kwargs::Vector{Pair{Symbol, Any}}

    function InvertGMRes(operation::L; kwargs...) where {L<:AbstractMatrix}
        S = valtype(operation)
        if isempty(kwargs)
            return new{S, L}(operation, Pair{Symbol, Any}[])
        else
            return new{S, L}(operation, kwargs)
        end
    end
end

function Base.:(*)(m::InvertGMRes, v::AbstractVector)
    return gmres(m.operation, v; m.kwargs...)
end

function LinearAlgebra.mul!(y::AbstractVector{S}, m::InvertGMRes, x::AbstractVector) where {S}
    fill!(y, zero(S))
    gmres!(y, m.operation, x; initially_zero=true, m.kwargs...)
end


function Invert(operation::L, algorithm::Symbol=:minres; kwargs...) where {L<:AbstractMatrix}
    S = valtype(operation)
    if algorithm == :minres
        T = InvertMinRes
    elseif algorithm == :gmres
        T = InvertGMRes
    else
        T = nothing
        throw(ArgumentError("unsupported algorithm $algorithm"))
    end
    return T(operation; kwargs...)
end
