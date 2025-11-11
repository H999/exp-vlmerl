# Experiment: Synergistic Co-Optimization of VLMs with ERL

The combination of [`vlmrm`](https://github.com/AlignmentResearch/vlmrm) and [`GeneticCNN`](https://github.com/H999/GeneticCNN-torch-test-with-base-model) in Simulated Robotics (MuJoCo) Envs.

Detail formula and explain [`idea`](idea.pdf)

## Quick Start

1. start docker

    ```bash
    docker compose up -d
    ```

2. go inside docker container and install

    ```bash
    bash ./install.bash
    ```

3. start the test

    ```bash
    python testeapopwithenv.py
    ```
