using Documenter, SimpleLinear

makedocs(;
    modules=[SimpleLinear],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/kyungminlee/SimpleLinear.jl/blob/{commit}{path}#L{line}",
    sitename="SimpleLinear.jl",
    authors="Kyungmin Lee",
    assets=String[],
)

deploydocs(;
    repo="github.com/kyungminlee/SimpleLinear.jl",
    devbranch="dev",
)
