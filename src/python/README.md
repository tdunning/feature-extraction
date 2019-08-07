# Python Feature Extraction Examples

## Methods Illustrated
Symbol combinations

Quantiles, log-odds and binning

Reduction to integers using ordinal encoding

One hot encoding (and counting)

Frequency encoding and unknown entity

Luduan features
## Use Cases
Web log
* Domain, referer, user agent
* referer + domain hashed encoding

Header fields
* ordering 
* language frequency
* language + charset combos
* unknown word

Purchase amount history
* log-odds, binning on purchase size
* symbol combination (store + quantile-bin)

Viewership
* time quadrature, one-hot
* one-hot time encodings

Common point of compromise
* Luduan

Energy models
* 5P model parameters
* residuals

Credit card gangs
* card velocity
