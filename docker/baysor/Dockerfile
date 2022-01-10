FROM julia:1.7.1
RUN apt-get update && apt-get install -y build-essential
## Jupyter
RUN apt-get install -y python3 python3-pip vim htop
RUN pip3 install jupyterlab numpy scipy matplotlib seaborn pandas sklearn scikit-image \
        jupyter_contrib_nbextensions jupyter_nbextensions_configurator jupyterthemes
RUN pip3 install -Iv six==1.12.0
RUN useradd -ms /bin/bash baysoruser
USER baysoruser
WORKDIR /home/baysoruser
RUN julia -e 'using Pkg; Pkg.add("IJulia"); Pkg.build(); using IJulia;'
RUN julia -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/kharchenkolab/Baysor.git")); Pkg.add(Pkg.PackageSpec(name="PackageCompiler", version="2.0.2"));'
RUN julia -e 'using Pkg; Pkg.build(); using Baysor'
RUN julia -e 'using Pkg; Pkg.add("MKL")'
RUN julia -e 'using PackageCompiler; import Baysor, Pkg; Pkg.activate(".."); Pkg.activate(dirname(dirname(pathof(Baysor)))); Pkg.instantiate(); create_sysimage(:Baysor; precompile_execution_file="$(dirname(pathof(Baysor)))/../bin/precompile.jl", sysimage_path="BaysorSysimage.so")'
USER root
RUN \
# Construct the script below
# https://docs.julialang.org/en/v1/manual/faq/#How-do-I-pass-options-to-julia-using-#!/usr/bin/env?
# https://docs.julialang.org/en/v1/manual/faq/#catch-ctrl-c
##!/usr/bin/env bash
#
##=
#exec julia -e 'include(popfirst!(ARGS))' --sysimage=/BaysorSysimage.so --color=yes --startup-file=no "${BASH_SOURCE[0]}" "$@"
#=#
#
#
#@show ARGS
#import Baysor
#Baysor.run_cli()
#
     #printf "#!/usr/bin/env julia --sysimage=/BaysorSysimage.so\n\nimport Baysor\nBaysor.run_cli()" >> /bin/baysor && \
     printf '#!/usr/bin/env bash\n\n#=\n\nexport HOME=/home/baysoruser\n\nexec julia -e "include(popfirst!(ARGS))" --sysimage=/home/baysoruser/BaysorSysimage.so --color=yes --startup-file=no "${BASH_SOURCE[0]}" "$@"\n\n=#\n\n@show ARGS\n\nimport Baysor\nBaysor.run_cli()' >> /bin/baysor && \
     chmod +x /bin/baysor
USER baysoruser
CMD ["/bin/bash"]
