![](media/MarioDemo.gif)


This code implements an agent that plays Super Mario Bros. The agent learns by reinforcement learning and Natural Evolution Strategy (NES) (https://openai.com/blog/evolution-strategies/) . It is an interesting fact that NES can be seen as a sort of approximation of the gradient using finite differences.

One nice feature is that the model's activations can be visualized. During training weights are stored on disk whenever a new top performing agent is discovered. The [visualize.py](visualize.py) script will use weights from disk to run an agent in the environment and show the activations of the model.

There is also a genetic algorithm implementation that works pretty pretty pretty good.

Some Observations :
* The model can learn enough to complete level 1.
* I think the agents are relying more on information from lstm layers and less on information from the convolutional layers.

Next steps :
* Visualize the activations of a recurrent convolution layer.