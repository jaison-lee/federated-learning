# Federated Learning & Data Privacy, 2025-2026

## Third Lab - 05 November 2025

Welcome to the third lab session of the Federated Learning & Data Privacy course! In our first two  labs, we ran experiments using our framework. Today, we will test a real production-based framework.


### EXERCISE 7: Get Started with Flower Framework
**Objective** Familiarize with [Flower](https://flower.ai/) and run your first Federated Simulation.

**Setup**:
Follow the [Get sterted with Flower](https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html) tutorial and launch your first Frederated Learning simulation. 

**Tip**: to create a new Python environment as indicated in the tutorial, you can first create and activate a new `conda` environment. To create a new environment use: 
```
conda create  -n <env_name> 
```
Then, to activate, use:
```
conda activate <env_name>
```
Now you are ready to proceed with Flower installation.

**Questions**:  
1. In the basic Flower setup described in the tutorial, what are the two main applications that the user must define? What are their respective roles?

$ \Rightarrow $ The two main applications must be defined are ServerApp and ClientApp:

- The ServerApp simulates what is running on the central server, it is responsible for orchestrating federated learning rounds, aggregating client update and deciding when training process stops. Additionaly, all implementation of strategies is also in this part to combine client parameters.

- The ClientApp simulates what is running on each participant. It is used to define how local training and evaluation are performed using client's dataset, then send results back to the server.


2. How do clients and server communicate and share parameters in Flower?Describe the object they use and the purpose of each specific field.

$ \Rightarrow $ Clients and server communicate by sending and receiving `Message` $object.

The `Message` object carries a `RecordDict` as the main payload. `RecordDict` contains 3 main record types:

- ArrayRecord: Contains model parameters as a dictionary of NumPy arrays
- MetricRecord: Contains training or evaluation metrics as a dictionary of integers, floats, lists of integers, or lists of floats.
- ConfigRecord: Contains configuration parameters as a dictionary of integers, floats, strings, booleans, or bytes. Lists of these types are also supported.

3. How can a user define how a client should perform training? Is there any constraint on the name of the training function?

$ \Rightarrow $ To define how a client perform training, user should implement local training steps in the client app under `train()` function and this function should be wrapped by a decorator `@app.train()`. In addition to the `@app.train()` decorator, there is no special constraint on the function's name. However, naming `train()` is usually considered as best practice for naming convention in this case. 

4. What is the difference between implementing an  `@app.evaluate` function on a server and on a client?

- On Client, the decorator @app.evaluate is used to register evaluation function with the client app which is to evaluate model performance on each individual client's data.(ref:[ClientApp](https://flower.ai/docs/framework/ref-api/flwr.client.ClientApp.html#flwr.client.ClientApp.evaluate)).

- However, accorrding to flower framework's document, there is no specific decorator app.evaluate for ServerApp(ref:[ServerApp](https://flower.ai/docs/framework/ref-api/flwr.server.ServerApp.html)). The evaluation function in ServerApp is registered by passing into evaluate_fn parameter of the strategy object (ref: [FedAvg](https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedAvg.html)). This function is used to evaluate performance of global model performance on centralized test dataset.

5. What is the purpose of the `Context` object? How can we modify the simulation parameters? 

$ \Rightarrow $ The `Context` object contains metadata and configuration about the current simulation such as hyperparameter, number of communication round, etc. and it is passed into client and server function to allow both of them retrieve current metadata for the simulation. These metadata is store in pyproject.toml where we can modify, for instance in this given code:

```
[tool.flwr.app.config]
num-server-rounds = 3
fraction-train = 0.5
local-epochs = 1
lr = 0.01

[tool.flwr.federations.local-simulation]
options.num-supernodes = 10
```



### EXERCISE 8: Tackle device Heterogeneity

**Objective** understand how to handle system heterogeneity in federated learning.

### Exercise 8.1: Handle device heterogeneity with FjORD

Review the paper [FjORD: Fair and Accurate Federated Learning under heterogeneous targets with Ordered Dropout](https://openreview.net/forum?id=4fLr7H5D_eT). The paper presents Ordered Dropout as a method to adapt federated learning to heterogenous setting.

**Preliminary Questions**: 
1. What is system heterogeneity and why it is a problem in Federated Learning?

$ \Rightarrow $ System heterogeneity is the diversity in the processing capabilities and network bandwidth of clients including hardware and computation (CPU, memory, etc), network connectivity, Power Constraints and workload (varies in computation load and data transmission speeds).

$ \Rightarrow $ System heterogeneity is a problem because it introduces several challenges in federated learning:
- The Straggler Effect and Timeouts: Different mobile hardware leads to significantly varying processing speeds. This disparity causes slower devices (stragglers) to delay the aggregation step in synchronous FL protocols, resulting in longer waits upon aggregation of updates.
- Infeasible Fixed Workload: Conventional FL methods, such as Federated Averaging (FedAvg), often mandate a uniform amount of local work (e.g., running the same number of local epochs, E) for all devices. System heterogeneity makes this unrealistic due to varying resource constraints.
- Device Exclusion and Training Bias: When devices cannot complete the mandatory work within a specified time window (due to low memory or slow processing), they are commonly dropped out of the procedure. Then only the fastest or most capable clients contribute updates, and the global model becomes biased toward the data distributions of those clients, reducing fairness and representativeness.
- Limited Global Model Capacity: The widely accepted norm in FL that local models must share the same architecture as the global model is problematic. To ensure that the least capable participants can complete the training, developers must limit the global model’s size. This restriction on model capacity leads to degraded accuracy.
- Exacerbated Instability: Dropping stragglers (as done in FedAvg) implicitly increases statistical heterogeneity and can adversely impact the convergence behavior of the model.
2. What is Ordered Dropout? 

$ \Rightarrow $ 

3. How does the aggregation rule account for device heterogeneity?

$ \Rightarrow $ 

**Implementation**: Follow the tutorial available at [Flower documentation](https://flower.ai/docs/baselines/fjord.html) and reproduce the results. Note that the tutorial reproduces the result for three different seeds. If you want to reduce the computational time, modify the `run.sh` script to use only one seed.

**IMPORTANT**: The tutorial uses an older version of Flower, different from the one used in the previous exercise. To reproduce the experiments, first create a new environment following the tutorial [Use Baselines](https://flower.ai/docs/baselines/how-to-use-baselines.html), and then, try to reproduce the experiment.


**Analysis**: Plot the results.   

![ResNet18-CIFAR10-500 global rounds](/TP3/flower-fjord/restnet18-cifar10.png)
![ResNet18-CIFAR10-FjORD](/TP3/flower-fjord/fjord.png)
![ResNet18-CIFAR10-FjORD w/KD](/TP3/flower-fjord/w-fjord.png)

1. Which one of the two implementation (with and without knowledge distillation) works better? Why?

$ \Rightarrow $ In general, The performance of FjORD with knowledge distillation (KD) is supposed to be better than the implementation without KD, especially for larger submodels. However, it cannot be seen clearly from the evaluation graph because it is not evaluated with best seed (The seed should be 124 according to result from Flower.ai). 

2. How do different values of p impact the model’s accuracy? Motivate your answer.

By Ordered Dropout(OD) mechanism, OD orders knowledge representation in nested submodels. Since the training is performed on these nested structures, the higher capacity submodels (higher p) capture and aggregate more knowledge from the network. Then, a larger p means fewer units are pruned, resulting in higher FLOPs and more parameters contributing to the forward and backward passes, thereby increasing the model's accuracy.

### BONUS EXERCISE Federated  Distillation

**Objective** implement Federated Distillation strategy described in [Communication-Efficient On-Device Machine Learning: Federated Distillation and Augmentation under Non-IID Private Data](https://arxiv.org/pdf/1811.11479). Reproduce and test only the Federated Distillation strategy, without Federated Augmentation.

**Tips**: 
- Flower automatically handles data communication between `ClientApp` and `ServerApp`. Focus on designing a custom `Message` and add a field in the `RecordDict` objects containing the logits of the client. Check [Communicate custom Messages](https://flower.ai/docs/framework/tutorial-series-customize-the-client-pytorch.html) to see how to design custmoized messages.
- Define a personalized strategy to implement the ensembling procedure in the `ServerApp`.  You can find some references here: [Flower Strategy Abstraction](https://flower.ai/docs/framework/explanation-flower-strategy-abstraction.html), [Customize a Flower Strategy](https://flower.ai/docs/framework/tutorial-series-build-a-strategy-from-scratch-pytorch.html).

---
At the end of the lesson, you can send your answers and code to: [francesco.diana@inria.fr](mailto:francesco.diana@inria.fr)

**IMPORTANT: you have time until 24/11/2025 to send me your solution. Late answers will be penalized.**

