# Install FedML and Prepare the Distributed Environment
```
pip install fedml
```


# Run the example (step by step APIs)
```
sh run_step_by_step_example.sh 5 config/mnist_lr/fedml_config.yaml

sh run_step_by_step_example.sh 5 config/mnist_lr/fedml_config_topo.yaml
```


- enable_parameter_estimation=True: estimate time, convergence params every comm round
- group_comm_round=0: trigger dynamic group communication