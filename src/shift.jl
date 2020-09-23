export Shift

import IterativeSolvers
import LinearAlgebra

struct Shift{S<:Number, LinOp<:AbstractMatrix} <: AbstractMatrix{S}
    shift::S
    operation::LinOp
    function Shift(shift::S, operation::L) where {S<:Number, L<:AbstractMatrix}
        S2 = promote_type(S, valtype(operation))
        new{S2, L}(convert(S2, shift), operation)
    end
end

function Base.:(*)(m::Shift, x::AbstractVector)
    return (m.operation * x) .+ (m.shift .* x)
end

function LinearAlgebra.mul!(y::AbstractVector, m::Shift, x::AbstractVector)
    mul!(y, m.operation, x)
    y .+= m.shift .* x
    return y
end

Base.size(s::Shift) = size(s.operation)

function Base.show(context::IOContext, mime::MIME, m::Shift)
    Base.show(context, mime, "Shift ($(m.shift)) of ")
    Base.show(context, mime, m.operation)
end
