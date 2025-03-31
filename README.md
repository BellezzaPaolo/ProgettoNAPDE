# ProgettoNAPDE- Numerical acceleration of neural network training processes
The aim of this project is to accelerate the training process of a fully connected neural network. Although the training process is central to their performance, it remains computationally intensive due to the iterative optimisation required to minimise prediction errors. Traditional methods, such as gradient descent, offer reduced computational cost per iteration, but often suffer from slower convergence. Conversely, parallel-in-time algorithms such as the Parareal method, originally developed to solve ordinary differential equations (ODEs), promise faster processing through parallel execution. In this work, we explore the numerical acceleration of neural network training by combining stochastic gradient descent with the parallel-in-time algorithm. Our approach exploits the coarse-to-fine parallel updating mechanism of the parareal method to increase the convergence speed of (Stochastic) Gradient Descent. Through extensive numerical experiments, we study the practical implications of this integration, including potential trade-offs between accuracy and speed. This hybrid approach redefines the computational limits of large-scale neural network training. The interplay between stochastic updating and parallel refinement affects stability and performance.

## Installation and Usage
The installation is straighforward, only a clone of the repository is sufficient:
```bash
git clone https://github.com/BellezzaPaolo/ProgettoNAPDE-MacchiniBellezza.git
```
To use it set the parameters of the network and optimizers in the file ./Code/Dati.m and run the ./Code/main.m file. The dipendences with the subfolder are handled in the first lines of the main file.
