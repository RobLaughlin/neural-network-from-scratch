# Backpropogation From "Scratch" With Python & NumPy
This implementation uses a feed-forward network with backpropigation and stochastic gradient descent,
and the backpropogation algorithm is implemented in Python using matrix operations provided by NumPy.  
  
The demo network was trained on the [MNIST Database](http://yann.lecun.com/exdb/mnist/), a collection of 70,000 28x28 images of handwritten
digits in the range [0, 9]. Using just the basic sigmoid function, backpropogation, and stochastic gradient descent (SGD),
the network was able to correctly identify roughly (95~96)% of the test handwritten images included in the MNIST DB.  
  
A demo.py file is included to see if the network can correctly identify hand-**drawn** images being trained on **handwritten** images.
It seems to perform moderately well, although there are of course many differences between computer-generated paints
and actual handwritten digits, so the original 96% mark is a bit of a stretch using the tkinter canvas.  
  
Included is an image extraction script, extract_images.py which will pull all of the image data from the MNIST dataset
into .PNG files to get an idea of what the network is looking at. This was used to build the demo.py file for example.
Additionally, there is a learn_cos_curve.py file which demonstrates how SGD is applied to a static dataset.  
  
## Demo.py using trained MNIST data
![Demo.py Example](https://i.imgur.com/4lHeCcg.gif)

## learn_cos_curve.py (Example of SGD on static dataset)
![Example of SGD on static dataset](https://i.imgur.com/3SZ00Ls.gif)

## Helpful resources that also use the MNIST DB

* [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) (Free online book)
* [3Blue1Brown Playlist on Machine Learning](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
