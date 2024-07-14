***DESIGN PATTERN AND PROGRAMMING PARADIGM***

    1. Design Pattern - the Python file have a corresponding Jupyter Notebook file, so an interactive jupyter
                        notebook code layout has been adopted
    
    2. Software Paradigm - a Functional paradigm has been adopted
    
    3. Work in progress - the code has been developed merely to show case coding abilities, nor is it designed
                          according to software beased parctices.

***THE HESTON MODEL***
  
      1. Resources - https://arxiv.org/pdf/1511.08718.pdf, Full and Fast Calibration of the Heston Stochastic
         Volatility model by Yiran Cui et al

      2. Resources - Options, Futures and other Derivatives by John C Hull
  
      3. Resources - The Heston model and its Extensions in Matlab and C# by Fabrice Douglas Rouah

      4. Resources - Numerical Optimization by Jorge Nocebal and Stephen J. Wright

***We present functions (for ease of unit testing) and algorithms in python for various problems for the development of the
Heston model.***

***DEVELOPMENT STEPS***
  
       1. Libraries
       2. Settings
  
***HESTON MODEL SEMI-CLOSED FORM CALL OPTION PRICE***

        1. We present different but equivalent forms of the Heston model Characteristic functions with their
           respective properties:

                1.1 Shoutens
                1.2 Albretcher et al
                1.3 Yiran Cui et al
                1.4 Original derivation - Discontinuities, Oscillations and Undefined at u=0

        2. Yiran Cui and Shoutens Characteristic function

              2.1 Heston model Call price by numerical integration and integrand defined as per above
              characteristics functions:

                      2.1.1 Quadrature methods - Yiran Cui and Shoutens formualtion

                                 2.1.1.1 Yiran Cui and Shoutens Complex Integrands functions
                                 2.1.1.2 Yiran Cui and Shoutens Heston Call Option price

         3. Original and Albretcher Heston Characteristic formulation

              3.1 Heston model Call price by numerical integration and integrand defined as per above characteristics
                  functions:

                    3.1.1 Quadrature methods - Original and Albretcher formulation

                                  3.1.1.1 Original and Albretcher Complex Integrands functions
                                  3.1.1.2 Original and Albretcher Heston Option Call price

         4. Heston model Call price by Fourier transforms and numerical integration:

                4.1. Carr and Mardan and Fast Fourier Transforms
                4.2. Lewis fundamental transforms
                4.3. Carr and Mardan damping factor alpha
                4.4. Truncating the domain of integration
  
  ***EULER-MILLSTEIN MONTE CARLO SIMULATION HESTON MODEL***
   
          1. Heston model Call price by simulation:

                    1.1. Euler - Millstein simulation
                    1.2. Simulation plot
  
  ***HESTON MODEL CALIBRATION***
    
          1. Options current prices download and data cleaning
          2. The Least squares Error Function
          3. Choosing initial starting points
          4. Optimization methods -- Gradient and Gradient-free:

                     4.1 Sequantial Linear Quadratic Programming: Gradient

                     4.2 BFGS: Gradient

                     4.3 Gauss Newton - algorithm presented

                     4.4 Nelder Mead - Gradient-free

                     4.5 Differential Evolution - Gradient-free

                     4.6 Heston Error Function for Gradient free methods

                     4.7 Simulated anealing - algorithm presented
    
         5. Gauss Newton Algorithm, Gradient Descent and Levenberg Marquardt

                5.1 Here we present the Gauss Newton algorithm:

                      5.1.1. Heston Error Jacobian and Hessian Function -- via Finite Difference method
                      5.1.2. Heston Error Residuals Function -- via Finite Difference method
                      5.1.3. Heston Error Gradient Function
                      5.1.4. An algorithm

                 We also present Gradient Descent and Levenberg Marquardt
      
  ***HESTON MODEL IMPLIED VOLATILITY***
       
           1. The Heston model implied volatility estimation via:

                  1.1  Bisection method algorithm
                  1.2. Gradient Descent algorithm
     
  ***HEDGING***
       
           1. Heston hedging ratios via Finite Difference method:

                1.1  Delta hedging
                1.2. Delta Sigma hedging
                1.3. Delta Sigma Gamma hedging
      
