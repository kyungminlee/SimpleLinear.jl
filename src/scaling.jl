using LinearAlgebra

export SizedScaling

import Base.size, Base.eltype
import Base.*, Base.-, Base.+, Base.*, Base.\, Base.//, Base.÷
import LinearAlgebra.rmul!, LinearAlgebra.lmul!, LinearAlgebra.mul!

import Base.transpose, Base.adjoint
import Base.real, Base.imag, Base.conj

import LinearAlgebra.RealHermSymComplexSym
import LinearAlgebra.RealHermSymComplexHerm
import LinearAlgebra.RealHermSym
import LinearAlgebra.AbstractTriangular
import LinearAlgebra.QRCompactWY


struct SizedScaling{T<:Number}<:AbstractMatrix{T}
    n::Int
    λ::T
    function SizedScaling{T}(n::Integer, λ::Number) where T
        @boundscheck begin
            n > 0 || throw(ArgumentError("size should be positive ($size)"))
        end
        return new{T}(n, λ)
    end
end

SizedScaling(n::Integer, λ::T) where T = SizedScaling{T}(n, λ)
SizedScaling(n::Integer, scale::LinearAlgebra.UniformScaling{T}) where T = SizedScaling{T}(n, scale.λ)
SizedScaling{T}(n::Integer, scale::LinearAlgebra.UniformScaling) where T = SizedScaling{T}(n, scale.λ)

SizedScaling(s::SizedScaling) = s
SizedScaling{T}(s::SizedScaling{T}) where T = s
SizedScaling{T}(s::SizedScaling) where T = SizedScaling{T}(size(s, 1), s.λ)

Base.eltype(::Type{SizedScaling{T}}) where T = T

Base.size(x::SizedScaling) = (x.n, x.n)
function Base.size(x::SizedScaling, d::Integer)
    if d<1
        throw(ArgumentError("dimension must be ≥ 1, got $d"))
    end
    return d<=2 ? s.n : 1
end

@inline function Base.getindex(x::SizedScaling{T}, i::Integer, j::Integer) where T
    @boundscheck checkbounds(x, i, j)
    return i == j ? x.λ : zero(T)
end

function Base.replace_in_print_matrix(A::SizedScaling,i::Integer,j::Integer,s::AbstractString)
    i==j ? s : Base.replace_with_centered_mark(s)
end

LinearAlgebra.ishermitian(s::SizedScaling{<:Real}) = true
LinearAlgebra.ishermitian(s::SizedScaling{<:Number}) = isreal(s.λ)
LinearAlgebra.issymmetric(s::SizedScaling{<:Number}) = true
LinearAlgebra.isposdef(s::SizedScaling{<:Number}) = isposdef(s.λ)

LinearAlgebra.factorize(s::SizedScaling) = s

Base.real(s::SizedScaling) = SizedScaling(size(s), real(s.λ))
Base.imag(s::SizedScaling) = SizedScaling(size(s), imag(s.λ))

Base.iszero(s::SizedScaling) = iszero(s.λ)
Base.isone(s::SizedScaling) = isone(s.λ)

LinearAlgebra.isdiag(s::SizedScaling) = true
LinearAlgebra.istriu(s::SizedScaling, k::Integer=0) = k <= 0 || iszero(s.λ) ? true : false
LinearAlgebra.istril(s::SizedScaling, k::Integer=0) = k >= 0 || iszero(s.λ) ? true : false

function LinearAlgebra.triu!(s::SizedScaling, k::Integer=0)
    n = size(s,1)
    if !(-n + 1 <= k <= n + 1)
        throw(ArgumentError(string("the requested diagonal, $k, must be at least ",
            "$(-n + 1) and at most $(n + 1) in an $n-by-$n matrix")))
    elseif k > 0
        s.λ = 0
    end
    return s
end

function LinearAlgebra.tril!(s::SizedScaling, k::Integer=0)
    n = size(s,1)
    if !(-n - 1 <= k <= n - 1)
        throw(ArgumentError(string("the requested diagonal, $k, must be at least ",
            "$(-n - 1) and at most $(n - 1) in an $n-by-$n matrix")))
    elseif k < 0
        s.λ = 0
    end
    return s
end

Base.:(==)(a::SizedScaling, b::SizedScaling) = (a.n == b.n) && (a.λ == b.λ)
Base.:(+)(a::SizedScaling) = a
Base.:(-)(a::SizedScaling) = SizedScaling(a.n, -a.λ)

function Base.:(+)(a::SizedScaling, b::SizedScaling)
    if a.n != b.n
        throw(DimensionMismatch("dimensions must match: a has dims $(size(a)), b has dims $(size(b))"))
    end
    return SizedScaling(a.n, a.λ + b.λ)
end

function Base.:(-)(a::SizedScaling, b::SizedScaling)
    if a.n != b.n
        throw(DimensionMismatch("dimensions must match: a has dims $(size(a)), b has dims $(size(b))"))
    end
    return SizedScaling(a.n, a.λ - b.λ)
end

for f in  (:+, :-)
    @eval function Base.$f(U::SizedScaling, D::Diagonal)
        return Diagonal($f(U.λ, D.diag))
    end
    @eval function Base.$f(D::Diagonal, U::SizedScaling, )
        return Diagonal($f(D.diag, U.λ))
    end
    @eval function Base.$f(U::SizedScaling, S::Symmetric)
        return Symmetric($f(U, S.data), sym_uplo(S.uplo))
    end
    @eval function Base.$f(S::Symmetric, U::SizedScaling)
        return Symmetric($f(S.data, U), sym_uplo(S.uplo))
    end
    @eval function Base.$f(U::SizedScaling{<:Real}, H::Hermitian)
        return Hermitian($f(U, H.data), sym_uplo(H.uplo))
    end
    @eval function Base.$f(H::Hermitian, U::SizedScaling{<:Real})
        return Hermitian($f(H.data, U), sym_uplo(H.uplo))
    end
end

Base.:(*)(b::Number, a::SizedScaling) = SizedScaling(a.n, a.λ * b)
Base.:(*)(a::SizedScaling, b::Number) = SizedScaling(a.n, a.λ * b)
Base.:(/)(a::SizedScaling, b::Number) = SizedScaling(a.n, a.λ / b)
Base.:(÷)(a::SizedScaling, b::Number) = SizedScaling(a.n, a.λ ÷ b)
Base.:(//)(a::SizedScaling, b::Number) = SizedScaling(a.n, a.λ // b)
Base.:(\)(b::Number, a::SizedScaling) = SizedScaling(a.n, b \ a.λ)

function Base.:(*)(a::SizedScaling, b::SizedScaling)
    a.n == b.n || throw(DimensionMismatch("dimensions must match: a has dims $(size(a)), b has dims $(size(b))"))
    return SizedScaling(a.n, a.λ * b.λ)
end

function Base.:(*)(a::SizedScaling, V::AbstractVector)
    a.n == length(V) || throw(DimensionMismatch("dimensions must match: a has dims $(size(a)), V has dims $(size(V))"))
    return a.λ .* V
end

function Base.:(*)(V::AbstractVector, a::SizedScaling)
    a.n == length(V) || throw(DimensionMismatch("dimensions must match: a has dims $(size(a)), V has dims $(size(V))"))
    return V .* a.λ
end

function Base.:(*)(A::LinearAlgebra.AbstractTriangular, U::SizedScaling)
    return rmul!(copyto!(similar(A, promote_op(*, eltype(A), eltype(U))), A), U)
end

function Base.:(*)(U::SizedScaling, B::LinearAlgebra.AbstractTriangular)
    return lmul!(U, copyto!(similar(B, promote_op(*, eltype(B), eltype(U))), B))
end

function Base.:(*)(A::AbstractMatrix, U::SizedScaling)
    return rmul!(copyto!(similar(A, promote_op(*, eltype(A), eltype(U)), size(A)), A), U)
end

function Base.:(*)(U::SizedScaling, A::AbstractMatrix)
    return lmul!(U, copyto!(similar(A, promote_op(*, eltype(A), eltype(U)), size(A)), A))
end


function LinearAlgebra.rmul!(A::AbstractMatrix, U::SizedScaling)
    require_one_based_indexing(A)
    size(A, 2) == size(U, 1) || throw(DimensionMismatch("dimensions must match: A has dims $(size(A)), U has dims $(size(U))"))
    A .= A .* U.λ
    return A
end


function LinearAlgebra.lmul!(U::SizedScaling, A::AbstractMatrix)
    require_one_based_indexing(A)
    size(U, 2) == size(A, 1) || throw(DimensionMismatch("dimensions must match: U has dims $(size(U)), A has dims $(size(A))"))
    A .= U.λ .* A
    return A
end

rmul!(A::Union{LowerTriangular,UpperTriangular}, U::SizedScaling) = typeof(A)(rmul!(A.data, U))
function rmul!(A::UnitLowerTriangular, U::SizedScaling)
    rmul!(A.data, U)
    for i = 1:size(A, 1)
        A.data[i,i] = U.λ
    end
    LowerTriangular(A.data)
end
function rmul!(A::UnitUpperTriangular, U::SizedScaling)
    rmul!(A.data, U)
    for i = 1:size(A, 1)
        A.data[i,i] = U.λ
    end
    UpperTriangular(A.data)
end

function lmul!(U::SizedScaling, B::UnitLowerTriangular)
    lmul!(U, B.data)
    for i = 1:size(B, 1)
        B.data[i,i] = U.λ
    end
    LowerTriangular(B.data)
end
function lmul!(U::SizedScaling, B::UnitUpperTriangular)
    lmul!(U, B.data)
    for i = 1:size(B, 1)
        B.data[i,i] = U.λ
    end
    UpperTriangular(B.data)
end

# TODO: >>>
# *(D::Adjoint{<:Any,<:SizedScaling}, B::SizedScaling) = Diagonal(adjoint.(D.parent.diag) .* B.diag)
# *(A::Adjoint{<:Any,<:AbstractTriangular}, U::SizedScaling) =
#     rmul!(copyto!(similar(A, promote_op(*, eltype(A), eltype(D.diag))), A), D)
# function *(adjA::Adjoint{<:Any,<:AbstractMatrix}, U::SizedScaling)
#     A = adjA.parent
#     Ac = similar(A, promote_op(*, eltype(A), eltype(D.diag)), (size(A, 2), size(A, 1)))
#     adjoint!(Ac, A)
#     rmul!(Ac, D)
# end

# *(D::Transpose{<:Any,<:SizedScaling}, B::SizedScaling) = Diagonal(transpose.(D.parent.diag) .* B.diag)
# *(A::Transpose{<:Any,<:AbstractTriangular}, U::SizedScaling) =
#     rmul!(copyto!(similar(A, promote_op(*, eltype(A), eltype(D.diag))), A), D)
# function *(transA::Transpose{<:Any,<:AbstractMatrix}, U::SizedScaling)
#     A = transA.parent
#     At = similar(A, promote_op(*, eltype(A), eltype(D.diag)), (size(A, 2), size(A, 1)))
#     transpose!(At, A)
#     rmul!(At, D)
# end

# *(U::SizedScaling, B::Adjoint{<:Any,<:Diagonal}) = Diagonal(D.diag .* adjoint.(B.parent.diag))
# *(U::SizedScaling, B::Adjoint{<:Any,<:AbstractTriangular}) =
#     lmul!(D, copyto!(similar(B, promote_op(*, eltype(B), eltype(D.diag))), B))
# *(U::SizedScaling, adjQ::Adjoint{<:Any,<:Union{QRCompactWYQ,QRPackedQ}}) = (Q = adjQ.parent; rmul!(Array(D), adjoint(Q)))
# function *(U::SizedScaling, adjA::Adjoint{<:Any,<:AbstractMatrix})
#     A = adjA.parent
#     Ac = similar(A, promote_op(*, eltype(A), eltype(D.diag)), (size(A, 2), size(A, 1)))
#     adjoint!(Ac, A)
#     lmul!(D, Ac)
# end

# *(U::SizedScaling, B::Transpose{<:Any,<:Diagonal}) = Diagonal(D.diag .* transpose.(B.parent.diag))
# *(U::SizedScaling, B::Transpose{<:Any,<:AbstractTriangular}) =
#     lmul!(D, copyto!(similar(B, promote_op(*, eltype(B), eltype(D.diag))), B))
# function *(U::SizedScaling, transA::Transpose{<:Any,<:AbstractMatrix})
#     A = transA.parent
#     At = similar(A, promote_op(*, eltype(A), eltype(D.diag)), (size(A, 2), size(A, 1)))
#     transpose!(At, A)
#     lmul!(D, At)
# end

# *(D::Adjoint{<:Any,<:Diagonal}, B::Adjoint{<:Any,<:Diagonal}) =
#     Diagonal(adjoint.(D.parent.diag) .* adjoint.(B.parent.diag))
# *(D::Transpose{<:Any,<:Diagonal}, B::Transpose{<:Any,<:Diagonal}) =
#     Diagonal(transpose.(D.parent.diag) .* transpose.(B.parent.diag))



function rmul!(A::SizedScaling, B::SizedScaling)
    if A.n != B.n
        throw(DimensionMismatch("dimensions must match: a has dims $(size(A)), b has dims $(size(B))"))
    end
    A.λ = A.λ * B.λ
    return A
end

function lmul!(A::SizedScaling, B::SizedScaling)
    if A.n != B.n
        throw(DimensionMismatch("dimensions must match: a has dims $(size(A)), b has dims $(size(B))"))
    end
    B.λ = A.λ * B.λ
    return B
end


function lmul!(adjA::Adjoint{<:Any,<:SizedScaling}, B::AbstractMatrix)
    A = adjA.parent
    return lmul!(adjoint(A), B)
end
function lmul!(transA::Transpose{<:Any,<:SizedScaling}, B::AbstractMatrix)
    A = transA.parent
    return lmul!(transpose(A), B)
end

function rmul!(A::AbstractMatrix, adjB::Adjoint{<:Any,<:SizedScaling})
    B = adjB.parent
    return rmul!(A, adjoint(B))
end
function rmul!(A::AbstractMatrix, transB::Transpose{<:Any,<:SizedScaling})
    B = transB.parent
    return rmul!(A, transpose(B))
end



@inline mul!(out::AbstractVector, A::SizedScaling, in::AbstractVector,
             alpha::Number, beta::Number) =
    out .= in .*ₛ (A.λ * alpha) .+ out .*ₛ beta
@inline mul!(out::AbstractVector, A::Adjoint{<:Any,<:SizedScaling}, in::AbstractVector,
             alpha::Number, beta::Number) =
    out .= in .*ₛ (conj(A.λ) * alpha) .+ out .*ₛ beta
@inline mul!(out::AbstractVector, A::Transpose{<:Any,<:SizedScaling}, in::AbstractVector,
             alpha::Number, beta::Number) =
    out .= in .*ₛ (A.parent.λ * alpha) .+ out .*ₛ beta

@inline mul!(out::AbstractMatrix, A::SizedScaling, in::StridedMatrix,
             alpha::Number, beta::Number) =
    out .= in .*ₛ (A.λ * alpha) .+ out .*ₛ beta
@inline mul!(out::AbstractMatrix, A::Adjoint{<:Any,<:SizedScaling}, in::StridedMatrix,
             alpha::Number, beta::Number) =
    out .= in .*ₛ (conj(A.λ) * alpha) .+ out .*ₛ beta
@inline mul!(out::AbstractMatrix, A::Transpose{<:Any,<:SizedScaling}, in::StridedMatrix,
             alpha::Number, beta::Number) =
    out .= in .*ₛ (A.parent.λ * alpha) .+ out .*ₛ beta


@inline mul!(out::AbstractMatrix, A::SizedScaling, in::Adjoint{<:Any,<:StridedMatrix},
             alpha::Number, beta::Number) =
    out .= in .*ₛ (A.λ * alpha) .+ out .*ₛ beta
@inline mul!(out::AbstractMatrix, A::Adjoint{<:Any,<:SizedScaling}, in::Adjoint{<:Any,<:StridedMatrix},
             alpha::Number, beta::Number) =
    out .= in .*ₛ (conj(A.λ) * alpha) .+ out .*ₛ beta
@inline mul!(out::AbstractMatrix, A::Transpose{<:Any,<:SizedScaling}, in::Adjoint{<:Any,<:StridedMatrix},
             alpha::Number, beta::Number) =
    out .= in .*ₛ (A.parent.λ * alpha) .+ out .*ₛ beta


@inline mul!(out::AbstractMatrix, A::SizedScaling, in::Transpose{<:Any,<:StridedMatrix},
             alpha::Number, beta::Number) =
    out .= in .*ₛ (A.λ * alpha) .+ out .*ₛ beta
@inline mul!(out::AbstractMatrix, A::Adjoint{<:Any,<:SizedScaling}, in::Transpose{<:Any,<:StridedMatrix},
             alpha::Number, beta::Number) =
    out .= in .*ₛ (conj(A.λ) * alpha) .+ out .*ₛ beta
@inline mul!(out::AbstractMatrix, A::Transpose{<:Any,<:SizedScaling}, in::Transpose{<:Any,<:StridedMatrix},
             alpha::Number, beta::Number) =
    out .= in .*ₛ (A.parent.λ * alpha) .+ out .*ₛ beta


@inline mul!(out::AbstractMatrix, in::StridedMatrix, A::SizedScaling,
             alpha::Number, beta::Number) =
    out .= in .*ₛ (A.λ * alpha) .+ out .*ₛ beta
@inline mul!(out::AbstractMatrix, in::StridedMatrix, A::Adjoint{<:Any,<:SizedScaling},
             alpha::Number, beta::Number) =
    out .= in .*ₛ (conj(A.λ) * alpha) .+ out .*ₛ beta
@inline mul!(out::AbstractMatrix, in::StridedMatrix, A::Transpose{<:Any,<:SizedScaling},
             alpha::Number, beta::Number) =
    out .= in .*ₛ (A.parent.λ * alpha) .+ out .*ₛ beta


@inline mul!(out::AbstractMatrix, in::Adjoint{<:Any,<:StridedMatrix}, A::SizedScaling,
             alpha::Number, beta::Number) =
    out .= in .*ₛ (A.λ * alpha) .+ out .*ₛ beta
@inline mul!(out::AbstractMatrix, in::Adjoint{<:Any,<:StridedMatrix}, A::Adjoint{<:Any,<:SizedScaling},
             alpha::Number, beta::Number) =
    out .= in .*ₛ (conj(A.λ) * alpha) .+ out .*ₛ beta
@inline mul!(out::AbstractMatrix, in::Adjoint{<:Any,<:StridedMatrix}, A::Transpose{<:Any,<:SizedScaling},
             alpha::Number, beta::Number) =
    out .= in .*ₛ (A.parent.λ * alpha) .+ out .*ₛ beta


@inline mul!(out::AbstractMatrix, in::Transpose{<:Any,<:StridedMatrix}, A::SizedScaling,
             alpha::Number, beta::Number) =
    out .= in .*ₛ (A.λ * alpha) .+ out .*ₛ beta
@inline mul!(out::AbstractMatrix, in::Transpose{<:Any,<:StridedMatrix}, A::Adjoint{<:Any,<:SizedScaling},
             alpha::Number, beta::Number) =
    out .= in .*ₛ (conj(A.λ) * alpha) .+ out .*ₛ beta
@inline mul!(out::AbstractMatrix, in::Transpose{<:Any,<:StridedMatrix}, A::Transpose{<:Any,<:SizedScaling},
             alpha::Number, beta::Number) =
    out .= in .*ₛ (A.parent.λ * alpha) .+ out .*ₛ beta




*(A::SizedScaling, transB::Transpose{<:Any,<:RealHermSymComplexSym}) = A * transB.parent
*(transA::Transpose{<:Any,<:RealHermSymComplexSym}, B::SizedScaling) = transA.parent * B
*(A::SizedScaling, adjB::Adjoint{<:Any,<:RealHermSymComplexHerm}) = A * adjB.parent
*(adjA::Adjoint{<:Any,<:RealHermSymComplexHerm}, B::SizedScaling) = adjA.parent * B
*(transA::Transpose{<:Any,<:RealHermSymComplexSym}, transD::Transpose{<:Any,<:SizedScaling}) = transA.parent * transD
*(transD::Transpose{<:Any,<:SizedScaling}, transA::Transpose{<:Any,<:RealHermSymComplexSym}) = transD * transA.parent
*(adjA::Adjoint{<:Any,<:RealHermSymComplexHerm}, adjD::Adjoint{<:Any,<:SizedScaling}) = adjA.parent * adjD
*(adjD::Adjoint{<:Any,<:SizedScaling}, adjA::Adjoint{<:Any,<:RealHermSymComplexHerm}) = adjD * adjA.parent

function mul!(C::AbstractMatrix, A::Adjoint{<:Any,<:SizedScaling}, B::Adjoint{<:Any,<:RealHermSymComplexSym})
    LinearAlgebra.check_A_mul_B!_sizes(C, A, B)
    return C .= conj(A.parent.λ) .* B
end
function mul!(C::AbstractMatrix, A::Transpose{<:Any,<:SizedScaling}, B::Transpose{<:Any,<:RealHermSymComplexHerm})
    LinearAlgebra.check_A_mul_B!_sizes(C, A, B)
    return C .= A.parent.λ .* B
end


@inline mul!(C::AbstractMatrix,
             A::Adjoint{<:Any,<:SizedScaling}, B::Adjoint{<:Any,<:RealHermSym},
             alpha::Number, beta::Number) = mul!(C, A, B.parent, alpha, beta)
@inline mul!(C::AbstractMatrix,
             A::Adjoint{<:Any,<:SizedScaling}, B::Adjoint{<:Any,<:RealHermSymComplexHerm},
             alpha::Number, beta::Number) = mul!(C, A, B.parent, alpha, beta)
@inline mul!(C::AbstractMatrix,
             A::Transpose{<:Any,<:SizedScaling}, B::Transpose{<:Any,<:RealHermSym},
             alpha::Number, beta::Number) = mul!(C, A, B.parent, alpha, beta)
@inline mul!(C::AbstractMatrix,
             A::Transpose{<:Any,<:SizedScaling}, B::Transpose{<:Any,<:RealHermSymComplexSym},
             alpha::Number, beta::Number) = mul!(C, A, B.parent, alpha, beta)

@inline mul!(C::AbstractMatrix,
             A::Adjoint{<:Any,<:SizedScaling}, B::Adjoint{<:Any,<:RealHermSymComplexSym},
             alpha::Number, beta::Number) =
    C .= B .*ₛ (conj(A.parent.λ) * alpha) .+ C .*ₛ beta
@inline mul!(C::AbstractMatrix,
             A::Transpose{<:Any,<:SizedScaling}, B::Transpose{<:Any,<:RealHermSymComplexHerm},
             alpha::Number, beta::Number) =
    C .= B .*ₛ (A.parent.λ * alpha) .+ C .*ₛ beta


function (/)(a::SizedScaling, b::SizedScaling)
    if a.n != b.n
        throw(DimensionMismatch("dimensions must match: a has dims $(size(a)), b has dims $(size(b))"))
    end
    return SizedScaling(a.n, a.λ / b.λ)
end

function ldiv!(s::SizedScaling{T}, v::AbstractVector{T}) where {T}
    if length(v) != length(s.n)
        throw(DimensionMismatch("scaling matrix matrix is $(s.n) by $(s.n) but right hand side has $(length(v)) rows"))
    end
    if iszero(s.λ)
        throw(SingularException())
    end
    v .= s.λ .\ v
    return v
end

function ldiv!(s::SizedScaling{T}, V::AbstractMatrix{T}) where {T}
    require_one_based_indexing(V)
    if size(V,1) != s.n
        throw(DimensionMismatch("scaling matrix matrix is $(s.n) by $(s.n) but right hand side has $(size(V,1)) rows"))
    end
    if iszero(s.λ)
        throw(SingularException(i))
    end
    V .= s.λ .\ V
    return V
end

function ldiv!(x::AbstractArray, A::SizedScaling, b::AbstractArray)
    return (x .= A.λ .\ b)
end

function ldiv!(adjS::Adjoint{<:Any,<:SizedScaling{T}}, B::AbstractVecOrMat{T}) where {T}
    return (S = adjS.parent; ldiv!(conj(S), B))
end

function ldiv!(transS::Transpose{<:Any,<:SizedScaling{T}}, B::AbstractVecOrMat{T}) where {T}
    return (S = transS.parent; ldiv!(S, B))
end

function ldiv!(S::SizedScaling, A::Union{LowerTriangular,UpperTriangular})
    broadcast!(\, parent(A), S.λ, parent(A))
    A
end


function rdiv!(A::AbstractMatrix{T}, S::SizedScaling{T}) where {T}
    require_one_based_indexing(A)
    m, n = size(A)
    if (k = size(S, 1)) ≠ n
        throw(DimensionMismatch("left hand side has $n columns but S is $k by $k"))
    end
    if iszero(S.λ)
        throw(SingularException(j))
    end
    @inbounds for j in 1:n
        ddj = dd[j]
        for i in 1:m
            A[i, j] /= S.λ
        end
    end
    A
end

function rdiv!(A::Union{LowerTriangular,UpperTriangular}, S::SizedScaling)
    broadcast!(/, parent(A), parent(A), S.λ)
    A
end

rdiv!(A::AbstractMatrix{T}, adjS::Adjoint{<:Any,<:SizedScaling{T}}) where {T} =
    (S = adjS.parent; rdiv!(A, conj(S)))
rdiv!(A::AbstractMatrix{T}, transS::Transpose{<:Any,<:SizedScaling{T}}) where {T} =
    (S = transS.parent; rdiv!(A, S))

(/)(A::Union{StridedMatrix, AbstractTriangular}, S::SizedScaling) =
    rdiv!((typeof(oneunit(eltype(S))/oneunit(eltype(A)))).(A), S)

(\)(F::Factorization, S::SizedScaling) =
    ldiv!(F, Matrix{typeof(oneunit(eltype(S))/oneunit(eltype(F)))}(S))
\(adjF::Adjoint{<:Any,<:Factorization}, S::SizedScaling) =
    (F = adjF.parent; ldiv!(adjoint(F), Matrix{typeof(oneunit(eltype(S))/oneunit(eltype(F)))}(S)))
(\)(A::Union{QR,QRCompactWY,QRPivoted}, B::SizedScaling) =
    invoke(\, Tuple{Union{QR,QRCompactWY,QRPivoted}, AbstractVecOrMat}, A, B)


conj(s::SizedScaling) = SizedScaling(size(s), conj(s.λ))
transpose(s::SizedScaling) = SizedScaling(reverse(size(s)), s.λ)
adjoint(s::SizedScaling) = SizedScaling(reverse(size(s)), conj(s.λ))


function diag(S::SizedScaling{T}, k::Integer=0) where {T}
    # every branch call similar(..., ::Int) to make sure the
    # same vector type is returned independent of k
    if k == 0
        return fill!(Vector{T}(undef, S.n), S.λ)
    elseif -size(S,1) <= k <= size(S,1)
        return fill!(Vector{T}(undef, S.n), 0)
    else
        throw(ArgumentError(string("requested diagonal, $k, must be at least $(-size(S, 1)) ",
            "and at most $(size(S, 2)) for an $(size(S, 1))-by-$(size(S, 2)) matrix")))
    end
end


tr(s::SizedScaling) = s.λ * s.n
det(s::SizedScaling) = s.λ^s.n
logdet(s::SizedScaling) = log(s.λ) * s.n

function logdet(s::SizedScaling{<:Complex}) # make sure branch cut is correct
    z = log(s.λ) * s.n
    complex(real(z), rem2pi(imag(z), RoundNearest))
end

# Matrix functions
for f in (:exp, :log, :sqrt,
    :cos, :sin, :tan, :csc, :sec, :cot,
    :cosh, :sinh, :tanh, :csch, :sech, :coth,
    :acos, :asin, :atan, :acsc, :asec, :acot,
    :acosh, :asinh, :atanh, :acsch, :asech, :acoth)
    @eval $f(S::SizedScaling) = SizedScaling(S.n, $f(S.λ))
end

function ldiv!(S::SizedScaling, B::StridedVecOrMat)
    m, n = size(B, 1), size(B, 2)
    if m != S.n
        throw(DimensionMismatch("sized scaling is $(S.n) by $(S.n) but right hand side has $m rows"))
    end
    (m == 0 || n == 0) && return B

    if iszero(S.λ)
        throw(SingularException(i))
    end
    B .= S.λ .\ B
    return B
end


(\)(S::SizedScaling, A::AbstractMatrix) =
    ldiv!(S, (typeof(oneunit(eltype(S))/oneunit(eltype(A)))).(A))

(\)(S::SizedScaling, b::AbstractVector) = S.diag .\ b

function (\)(Sa::SizedScaling, Sb::SizedScaling)
    if Sa.n != Sb.n
        throw(DimensionMismatch("A is $(Sa.n) by $(Sa.n) but B is $(Sb.n) by $(Sb.n)"))
    end
    return SizedScaling(Sa.n, Sa.λ .\ Sb.λ)
end

function inv(S::SizedScaling{T}) where T
    if iszero(S.λ)
        throw(SingularException(i))
    end
    return SizedScaling(S.n, inv(S.λ))
end

# TODO: pinv

function eigvals(S::SizedScaling{T}; permute::Bool=true, scale::Bool=true) where {T<:Number}
    return fill!(Vector{T}(undef, S.n), S.λ)
end

function eigvecs(S::SizedScaling{T}) where {T<:Number}
    return Matrix{T}(LinearAlgebra.I, (S.n, S.n))
end

function eigen(S::SizedScaling; permute::Bool=true, scale::Bool=true, sortby::Union{Function,Nothing}=nothing)
    if !isfinite(S.λ)
        throw(ArgumentError("matrix contains Infs or NaNs"))
    end
    Eigen(eigvals(D), eigvecs(D))
end


*(x::Adjoint{<:Any,<:AbstractVector},   S::SizedScaling) = Adjoint(map((t,s) -> t'*s, S.λ, parent(x)))
*(x::Transpose{<:Any,<:AbstractVector}, S::SizedScaling) = Transpose(map((t,s) -> transpose(t)*s, S.λ, parent(x)))
*(x::Adjoint{<:Any,<:AbstractVector},   S::SizedScaling, y::AbstractVector) = _mapreduce_prod(*, x, S, y)
*(x::Transpose{<:Any,<:AbstractVector}, S::SizedScaling, y::AbstractVector) = _mapreduce_prod(*, x, S, y)
dot(x::AbstractVector, S::SizedScaling, y::AbstractVector) = _mapreduce_prod(dot, x, S, y)

function _mapreduce_prod(f, x, S::SizedScaling, y)
    if isempty(x) && S.n == 0 && isempty(y)
        return zero(Base.promote_op(f, eltype(x), eltype(S), eltype(y)))
    else
        return mapreduce(t -> f(t[1], S.λ, t[3]), +, zip(x, y))
    end
end


# function Base.show(io::IO, mime::MIME, m::SizedScaling{T}) where {T}
#     Base.summary(io, m)
# end


function LinearAlgebra.mul!(y::AbstractVector, s::SizedScaling, x::AbstractVector)
    n, m = size(s)
    length(y) == n || throw(DimensionMismatch("y has length $(length(y)) and s has size $(size(s))"))
    length(x) == m || throw(DimensionMismatch("x has length $(length(x)) and s has size $(size(s))"))
    min_nm = min(n, m)
    y[1:min_nm] .= s.λ .* x[1:min_nm]
    y[min_nm+1:end] .= zero(eltype(y))
    return y
end


function LinearAlgebra.mul!(y::AbstractMatrix, s::SizedScaling, x::AbstractMatrix)
    check_A_mul_B!_sizes(y, s, x)
    n, m = size(s)
    min_nm = min(n, m)
    y[1:min_nm, :] .= s.λ .* x[1:min_nm, :]
    y[min_nm+1:end, :] .= zero(eltype(y))
    return y
end

function Base.:(*)(s::SizedScaling{Ts}, x::AbstractVecOrMat{Tv}) where {Ts, Tv}
    S = promote_type(Ts, Tv)
    y = Vector{S}(undef, size(s, 1))
    mul!(y, s, x)
    return y
end

function Base.:(*)(x::AbstractVecOrMat{Tv}, s::SizedScaling{Ts}) where {Ts, Tv}
    S = promote_type(Ts, Tv)
    y = Vector{S}(undef, size(s, 2))
    mul!(y, transpose(s), x)
    return y
end


function Base.:(^)(a::SizedScaling, b::Number)
    @boundscheck LinearAlgebra.checksquare(a)
    return SizedScaling(size(a), a.λ^b)
end
