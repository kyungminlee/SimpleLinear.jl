export Shift

import IterativeSolvers
import LinearAlgebra

struct Shift{S<:Number, L<:AbstractMatrix}<:AbstractMatrix{S}
    shift::S
    operation::L

    function Shift(shift::S, operation::L) where {S<:Number, L<:AbstractMatrix}
        S2 = promote_type(S, eltype(operation))
        new{S2, L}(convert(S2, shift), operation)
    end

    function Shift{S2}(shift::S, operation::L) where {S<:Number, L<:AbstractMatrix, S2}
        new{S2, L}(convert(S2, shift), operation)
    end
end

function Base.:(*)(m::Shift, x::AbstractVector)
    return (m.operation * x) .+ (m.shift .* x)
end

function LinearAlgebra.mul!(y::AbstractVector, m::Shift, x::AbstractVector)
    LinearAlgebra.mul!(y, m.operation, x)
    LinearAlgebra.BLAS.axpy!(m.shift, x, y)
    return y
end

Base.size(s::Shift, args...) = size(s.operation, args...)

function Base.show(io::IO, mime::MIME, m::Shift)
    Base.show(io, mime, "Shift ($(m.shift)) of ")
    Base.show(io, mime, m.operation)
end

function Base.display(s::Shift)
    println("Shift ($(s.shift)) of: ")
    display(s.operation)
end

# TODO: shift composition?
