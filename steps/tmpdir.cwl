#!/usr/bin/env cwl-runner
class: ExpressionTool
cwlVersion: v1.2

requirements:
  InlineJavascriptRequirement: {}
  ResourceRequirement:
    ramMin: 1000
    tmpdirMin: 1000
    outdirMin: 1000

expression: |
  ${
      const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
      const charactersLength = characters.length;
      let counter = 0;
      let result = "";
      while (counter < 10) {
        result += characters.charAt(Math.floor(Math.random() * charactersLength));
        counter += 1;
      }
      return {"tmp": result};
  }
inputs: []
outputs:
  tmp: string
