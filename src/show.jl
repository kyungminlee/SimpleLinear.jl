
function Base.show(io::IO, mime::MIME, m::Shift)
    Base.show(io, mime, "Shift ($(m.shift)) of ")
    Base.show(io, mime, m.operation)
end

function Base.display(s::Shift)
    println("Shift ($(s.shift)) of: ")
    display(s.operation)
end


function Base.show(io::IO, mime::MIME, m::FactorizeInvert)
    Base.show(io, mime, "FactorizeInvert of ")
    Base.show(io, mime, m.factorization)
end

function Base.display(s::FactorizeInvert)
    println("FactorizeInvert of:")
    display(s.factorization)
end


function Base.show(io::IO, mime::MIME, m::IterativeInvertMinRes)
    Base.show(io, mime, "IterativeInvertMinRes of ")
    Base.show(io, mime, m.operation)
end

function Base.display(s::IterativeInvertMinRes)
    println("IterativeInvertMinRes of: ")
    display(s.operation)
end



Base.size(s::IterativeInvertGMRes, args...) = size(s.operation, args...)

function Base.show(io::IO, mime::MIME, m::IterativeInvertGMRes)
    Base.show(io, mime, "IterativeInvertGMRes of ")
    Base.show(io, mime, m.operation)
end
