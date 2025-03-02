using LVModels
using Documenter

DocMeta.setdocmeta!(LVModels, :DocTestSetup, :(using LVModels); recursive=true)

makedocs(;
    modules=[LVModels],
    authors="Aditya Sahu <ge27zib@tum.de>",
    sitename="LVModels.jl",
    format=Documenter.HTML(;
        canonical="https://ge27zib.github.io/LVModels.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ge27zib/LVModels.jl",
    devbranch="master",
)
