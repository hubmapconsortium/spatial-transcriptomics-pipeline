#!/usr/bin/env cwl-runner

class: CommandLineTool
cwlVersion: v1.2
baseCommand: ["run_baysor.sh"]

requirements:
  DockerRequirement:
    dockerPull: vpetukhov/baysor:v0.5.0
  InitialWorkDirRequirement:
    listing:
      - entryname: run_baysor.sh
        entry: |-
          #!/usr/bin/env bash
          # Set the exit code of a pipeline to that of the rightmost command
          # to exit with a non-zero status, or zero if all commands of the pipeline exit
          set -o pipefail
          # cause a bash script to exit immediately when a command fails
          set -e
          # cause the bash shell to treat unset variables as an error and exit immediately
          set -u
          # echo each line of the script to stdout so we can see what is happening
          set -o xtrace
          #to turn off echo do 'set +o xtrace'
          # Call the baysor script that is processed by Julia and pass all arguments to it
          env
          cat /bin/baysor
          baysor run "$@"
  EnvVarRequirement:
    envDef:
      HOME: "/root"

inputs:
  csv:
    type: File
    inputBinding:
      position: 6
    doc: csv with transcript information
  priors:
    type: string?
    inputBinding:
      position: 7
    doc: Name of column with prior segmentation.
  x_col:
    type: string?
    inputBinding:
      position: 2
      prefix: -x
    default: xc
    doc: Name of the column with x information
  y_col:
    type: string?
    inputBinding:
      position: 3
      prefix: -y
    default: yc
    doc: Name of the column with y information
  z_col:
    type: string?
    inputBinding:
      position: 4
      prefix: -z
    default: zc
    doc: Name of the column with z information
  gene_col:
    type: string?
    inputBinding:
      position: 5
      prefix: --gene
    default: target
    doc: Name of the column with gene names
  scale:
    type: int?
    inputBinding:
      position: 1
      prefix: -s
    doc: Scale of pixels in image.

outputs:
  segmented:
    type: File[]
    outputBinding:
      glob: "segmentation*"

stdout: baysor_stdout.log
