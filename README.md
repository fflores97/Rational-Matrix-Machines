
# Introduction

This repository holds all code for the Rational Matrix Machine Project. The original idea by Pablo Ducru is to improve current methods used to fit poles to resonance data like VectFit (https://www.sintef.no/projectweb/vectfit/) using different regularizers to filter out overfitted poles.



# Dependencies

Our implementation uses Matlab code written by Vladimir Sobes. Since we didn't have immediate access to Matlab, we decided to implement the code on GNU Octave, which turned out to be quite compatible with this code.

* Python
    * Numpy
    * Matplotlib
    * Oct2Py for Octave

* Octave
    * `statistics` package

# References

* VectFit algorithm by B. Gustavsen and A. Semlyen (https://www.sintef.no/projectweb/vectfit/)
* Python implementation from PhilReinhold (https://github.com/PhilReinhold/vectfit_python)