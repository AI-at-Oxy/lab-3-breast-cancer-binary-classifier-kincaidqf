"""
For my comparison I will use a k-nearest neighbors classifier. It will use the same training and test datasets,
but the principle is slightly difference. A KNN classifier will look at the k (parameter) closest trianing samples
to the test sample and will predict which class the test sample belongs to based on the classes of those nearest
neighbors. Basically rather than deciding a binary label based on a linear decision (positive or negative side of
a sigmoid function), it will decide based on the classification of the k nearest comperable samples.

I chose KNN because I used a KNN measrument in my COMPS as part of my evaluation metrics to give me a classification
accuracy for real vs synthetic data. At the time I just used a stock model without fully understanding why I was using
it, since it wasn't a part of my actual experiment/investigation, and was just an evaluation metric. I wanted to take
this chance to actually get to understand the model better.
"""

