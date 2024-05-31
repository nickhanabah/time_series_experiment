# Time series modelling experiments

# Input

Let $X$ be a times series with $N$ features and a sequence lenght $S_{|X|}$. 
Let $Y$ be the corresponding target series of length $S_{|Y|}$, where $S_{|X|}$ equals $S_{|Y|}$. 

Now our modelling task consists of forecasting a future sequence of length $T$ regarding the target variable $Y$ based on a lookback window $L$
with regard to our feature variables $X_{i-1-L:i-1}$ and the past $Y_{i-1-L:i-1}$ values of our target variable. 

The aim is to find a function $f(.)$ that maps the input $Y_{i-1-L:i-1}$ and $X_{i-1-L:i-1}$ to the respective output $Y_{i:i+T}$. 

$Y_{i:i+T} = f(X_{i-1-L:i-1}, Y_{i-1-L:i-1})$


The proposed Dlinear model consists of a decomposition layer, where the input $Y_{i-1-L:i-1}$ and $X_{i-1-L:i-1}$ is decomposed into a seasonal and a trend components plus two linear layers to transform each of the decomposed inputs into some meaningful signal regarding the output $Y_{i:i+T}$. 

The operations are as follows: 


1. We combine the target and the feature variable with regards to our lookback window $i-1-L:i-1$ to a new feature matrix $\tilde{X}_{i-1-L:i-1}$. 

2. We decompose $\tilde{X}_{i-1-L:i-1}$ into its trend and seasonal component


$X_{trend_{i-1-L:i-1}}, X_{seasonal_{i-1-L:i-1}} = decomposition(X_{i-1-L:i-1})$ 

$Y_{hat} = W_{trend}^T X_{trend} + W_{seasonal}^T X_{seasonal}$