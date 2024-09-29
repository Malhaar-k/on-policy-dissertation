# MDPO with SMAC

- This is a fork of "marlbench/on-policy". I attempted the implementation of "Mirror Descent Policy Optimisation" on the StarCraft Multi Agent Challenge. 

- This was my dissertation project for the Master's in Advanced Control and Systems Engineering course I did at the University of Manchester.

- All training runs were executed on the university's computer cluster which uses Nvidia v100 GPUs. The container image used for training can be found [here]{https://hub.docker.com/r/malhaark/mappo-test/tags}. You could also pull the exact image used with the following:
```
    docker pull malhaark/mappo-test:0.4
```

### Note:
This project takes inspiration from [this]{https://arxiv.org/abs/2005.09814} paper to develop its policy optimisation algorithm with a few tweaks. 