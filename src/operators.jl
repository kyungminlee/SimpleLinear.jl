import LinearAlgebra

export MatrixPlusExpr
export MatrixTimesExpr




struct MatrixPlusExpr{S, Ops<:Tuple{Vararg{<:AbstractMatrix}}} <: AbstractMatrix{S}
    size::Tuple{Int, Int}
    terms::Ops
    function MatrixPlusExpr(args::AbstractMatrix ...)
        S = promote_type(map(eltype, args)...)
        T = typeof(args)
        n = length(args)
        n == 0 && throw(ArgumentError("number of arguments is zero"))
        for i in 2:n
            if size(args[i]) != size(args[1])
                throw(ArgumentError("sizes of the argument $(i-1) $(size(args[i])) and the argument 1 $(size(args[1])) do not match"))
            end
        end
        return new{S, T}(size(args[1]), args)
    end
end


struct MatrixTimesExpr{S, Ops<:Tuple{Vararg{<:AbstractMatrix}}} <: AbstractMatrix{S}
    size::Tuple{Int, Int}
    factors::Ops
    function MatrixTimesExpr(args::AbstractMatrix ...)
        S = promote_type(map(eltype, args)...)
        T = typeof(args)
        n = length(args)
        n == 0 && throw(ArgumentError("number of arguments is zero"))
        nrow = size(args[1], 1)
        m = size(args[1], 2)
        for i in 2:n
            if m != size(args[i], 1)
                throw(ArgumentError("sizes of the argument $(i-1) $(size(args[i-1])) and argument $i $(size(args[i])) do not match"))
            end
            m = size(args[i], 2)
        end
        ncol = m
        return new{S, T}((nrow, ncol), args)
    end
end


Base.size(x::MatrixPlusExpr) = x.size
Base.size(x::MatrixPlusExpr, i::Integer) = x.size[i]

Base.size(x::MatrixTimesExpr) = x.size
Base.size(x::MatrixTimesExpr, i::Integer) = x.size[i]

Base.eltype(::Type{MatrixPlusExpr{S, T}}) where {S, T} = S
Base.eltype(::Type{MatrixTimesExpr{S, T}}) where {S, T} = S


x = [1.0 0.0; 0.0 1.0]
y = [1.0 0.0; 0.0 2.0]

x = [1.0 0.0 0.0; 0.0 1.0 0.0]
y = [1.0 0.0 0.0; 0.0 2.0 0.0]
