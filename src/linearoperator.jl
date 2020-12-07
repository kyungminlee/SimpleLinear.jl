import LinearAlgebra

export LinearOperator

# === IterativeInvert ===

struct LinearOperator{
    S<:Number,
    A<:Function
}<:AbstractMatrix{S}
    size::Tuple{Int, Int}
    operation!::A
    function LinearOperator(::Type{S}, size::Tuple{<:Integer, <:Integer}, operation!::A) where {S<:Number, A<:Function}
        return new{S, A}(size, operation!)
    end
end

function LinearAlgebra.mul!(y::AbstractVector{So}, m::LinearOperator{S, A}, x::AbstractVector{Si}) where {So, Si, S, A}
    m.operation!(y, x)
    return y
end

Base.size(s::LinearOperator) = s.size
Base.size(s::LinearOperator, i::Integer) = s.size[i]
Base.eltype(::Type{LinearOperator{S, A}}) where {S, A} = S

function Base.show(io::IO, mime::MIME, m::LinearOperator)
    Base.show(io, mime, "LinearOperator with $(m.operation!)")
end
