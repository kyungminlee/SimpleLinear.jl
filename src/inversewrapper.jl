import LinearAlgebra
import SuiteSparse

# iterative solvers
export wrap_factorizedinvert!
export wrap_shift!
export wrap_cg!
export wrap_minres!
export wrap_chebyshev!
export wrap_bicgstabl!
export wrap_idrs!
export wrap_gmres!
export wrap_lsmr!
export wrap_lsqr!
# export wrap_jacobi!

# # workaround for missing ldiv!
function LinearAlgebra.ldiv!(A::AbstractVector, F::SuiteSparse.SPQR.QRSparse, B::AbstractVector)
    A[:] .= F \ B
    return A
end


function wrap_factorizedinvert!(f::LinearAlgebra.Factorization{T}) where {T}
    return function (y::VecOrMat{T}, x::AbstractVecOrMat{T})
        fill!(y, zero(eltype(y)))
        LinearAlgebra.ldiv!(y, f, x)
        return y
    end

    # F = typeof(f)
    # @show F

    # if hasmethod(LinearAlgebra.ldiv!, Tuple{VecOrMat{T}, F, AbstractVecOrMat{T}})
    #     @show "Bing1"
    #     return function (y::VecOrMat{T}, x::AbstractVecOrMat{T})
    #         fill!(y, zero(eltype(y)))
    #         LinearAlgebra.ldiv!(y, f, x)
    #         return y
    #     end
    # elseif hasmethod(LinearAlgebra.ldiv!, Tuple{F, VecOrMat{T}})
    #     @show "Bing2"
    #     return function (y::VecOrMat{T}, x::AbstractVecOrMat{T})
    #         copy!(y, x)
    #         LinearAlgebra.ldiv!(f, y)
    #         return y
    #     end
    # elseif hasmethod(Base.:(\), Tuple{VecOrMat{T}, F, AbstractVecOrMat{T}})
    #     @show "Bing3"
    #     return function (y::VecOrMat{T}, x::AbstractVecOrMat{T})
    #         copy!(y, f \ x)
    #         return y
    #     end
    # else
    #     throw(ArgumentError("type $F does not affect ldiv!(y, A, x), ldiv!(y, x) or Base.:(\\)"))
    # end
end


# function wrap_shift!(s::Number)
#     function _shift!(y::AbstractVecOrMat, x::AbstractVecOrMat)
#         y .= s .* x
#         return y
#     end
#     return _shift!
# end


function wrap_cg!(A::AbstractMatrix; kwargs...)
    function _cg!(x, b)
        fill!(x, zero(eltype(x)))
        result, history = IterativeSolvers.cg!(x, A, b; initially_zero=true, log=true, kwargs...)
        #history.resnom[end] > history.tol || @warn "cg! not converged: $history"
        history.isconverged || @warn "cg! not converged: $history"
        return x
    end
end


function wrap_minres!(A::AbstractMatrix; kwargs...)
    function _minres!(x, b)
        fill!(x, zero(eltype(x)))
        result, history = IterativeSolvers.minres!(x, A, b; initially_zero=true, log=true, kwargs...)
        history.isconverged || @warn "minres! not converged: $history"
        return x
    end
end


function wrap_chebyshev!(A::AbstractMatrix, λmin::Real, λmax::Real; kwargs...)
    function _chebyshev!(x, b)
        fill!(x, zero(eltype(x)))
        result, history = IterativeSolvers.chebyshev!(x, A, b, λmin, λmax; initially_zero=true, log=true, kwargs...)
        history.isconverged || @warn "chebyshev! not converged: $history"
        return x
    end
end

function wrap_bicgstabl!(A::AbstractMatrix, l=2; kwargs...)
    function _bicgstabl!(x, b)
        fill!(x, zero(eltype(x)))
        result, history = IterativeSolvers.bicgstabl!(x, A, b, l; log=true, kwargs...)
        history.isconverged || @warn "bicgstabl! not converged: $history"
        return x
    end
end


function wrap_idrs!(A::AbstractMatrix; s=8, kwargs...)
    function _idrs!(x, b)
        fill!(x, zero(eltype(x)))
        result, history = IterativeSolvers.idrs!(x, A, b; s=8, log=true, kwargs...)
        history.isconverged || @warn "idrs! not converged: $history"
        return x
    end
end


function wrap_gmres!(A::AbstractMatrix; kwargs...)
    function _gmres!(x, b)
        fill!(x, zero(eltype(x)))
        result, history = IterativeSolvers.gmres!(x, A, b; initially_zero=true, log=true, kwargs...)
        history.isconverged || @warn "gmres! not converged: $history"
        return x
    end
end


function wrap_lsmr!(A::AbstractMatrix; kwargs...)
    function _lsmr!(x, b)
        fill!(x, zero(eltype(x)))
        result, history = IterativeSolvers.lsmr!(x, A, b; log=true, kwargs...)
        history.isconverged || @warn "lsmr! not converged: $history"
        return x
    end
end


function wrap_lsqr!(A::AbstractMatrix; kwargs...)
    function _lsqr!(x, b)
        fill!(x, zero(eltype(x)))
        result, history = IterativeSolvers.lsqr!(x, A, b; log=true, kwargs...)
        history.isconverged || @warn "lsqr! not converged: $history"
        return x
    end
end


# function wrap_jacobi!(;kwargs...)
#     function _jacobi!(x, A, b)
#         fill!(x, zero(eltype(x)))
#         IterativeSolvers.jacobi!(x, A, b; kwargs...)
#         return x
#     end
# end
