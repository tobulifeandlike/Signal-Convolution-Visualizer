# Signal-Convolution-Visualizer
Using Claude AI to program a class of functions in Python Environment.
A demo of Digital Signal Processing class visualizing the convolution process of two signals. User can define two 1-D finite signals and see how their convolution is realized in time domain. 
Both signals starts at time index n0 = 0. 
Development path: theoretically know how two signals convolve each other. The process can be decomposed to four big steps: 1) Signal definition: two finite signals x[n] and h[n]; 2) Convolution parameter preparation: compute signals x[k] and h[n-k], x[k] is fixed on time axis and h[n-k] will shift as time variable n changes, then convolution values are computed with related functions; 3) Figure setup and UI widget construction: prepare spaces to draw diagram of the two signals and their convolution result y[n] at each time step; 4) Animation play.     
