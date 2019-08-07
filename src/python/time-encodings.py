"""
This demonstrates some simple ways to encode time so that models can
make sense of it.

The problem at hand is prediction of web traffic on various wikipedia pages.

The features we will use include:

* Lagged values for traffic
* Time of day expressed as continuous variables
* Day of week expressed as continuous variables
* Day of week expressed as one-hot variables
* Page URL
"""
import handout
import os

os.mkdir("handouts") # handout: exclude

doc = handout.Handout("handouts/time") # handout: exclude

doc.show() # handout: exclude