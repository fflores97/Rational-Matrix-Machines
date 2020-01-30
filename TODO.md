# Questions

* Is the algorithm limited to rational functions of denominator order 1?
* Is there a type in the original vectfit paper? Should eq. A.1 read

$$\left( \sum_{n+1}^{N} \frac{c_n}{s - a_n} +d +sh \right)-f(s)\left( \sum_{n-1}^N \frac{\tilde{c_n}}{s-a_n}\right ) = f(s)$$

instead of 

$$\left( \sum_{n+1}^{N} \frac{c_n}{s - a_n} +d +sh \right)-\left( \sum_{n-1}^N \frac{\tilde{c_n}}{s-a_n}\right ) = f(s)$$

?

# TODO

* Noise generated data
    * Decide on type of noise (most likely Gaussian)
    * Homo and heteroscedastic cases
* Test VectFit3 on generated data
* Code vector fitting implementation as Least-Squres problem (in-house implementation will circumvent some of the engineering tricks used in VectFit3)
* Add regularizer to VectFit
    * Decide on optimizer
* Code new one-step solution, including pole finding and regularizer filtering in one Least-Squares step

# TODO (2020/01/05)

* 
