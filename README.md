# AttackIntentInference
Attack Intent Inference of Hypersonic Glide Vehicle based on a Unified Dynamics and Decision-making Model

# Overall Architecture  
This repository implements a unified framework for inferring the most probable attack target of a Hypersonic Glide Vehicle (HGV). The framework combines dynamics-based trajectory propagation and decision-making models under uncertainty.

![algorithm](https://github.com/user-attachments/assets/84b62b59-a08d-402c-8fda-4b356cf9e8c7)

## 📝 Citation
**Y. Nam**, **H. Lee**, **H. Choi**, **W.-S. Ra**, and **C. Kwon**,  
"Attack Intent Inference of Hypersonic Glide Vehicle based on a Unified Dynamics and Decision-Making Model"  (**under review**)

## 📂 Project Structure

| File | Description |
|------|-------------|
| `main.py` | Running simulation episodes |
| `inference.py` | Bayesian approach-based attack intent inference  |
| `ckf.py` | Implementation of Cubature Kalman Filter |
| `model.py` | Physical dynamics model of the HGV |
| `model_jax.py` | JAX-based trajectory propagation for dynamics-based prediction |
| `plot.py` | 3D visualization utilities for trajectory |
| `utils.py` | Helper functions and parameters|

## ⚙️ Environment Requirements
Tested on:

- Python: 3.10.16
- CUDA 12, cuDNN 9.8
- GPU Acceleration: JAX with `jax-cuda12-pjrt`

To replicate the environment:

```bash
conda env create -f intent.yaml
conda activate intent
```

## 🚀 How to Run

```bash
python main.py
```

This command runs user-defined independent simulations, each with randomized threat levels and initial states for the Hypersonic Glide Vehicle (HGV).

The number of iterations is controlled by the loop at the end of main.py (e.g., for itr in range(20)).

The results are saved only when the variable "save_result = True".

Each output is stored as .npz files in the specified directory ./[FolderName]/, where FolderName is passed as an argument in test (foldername='Result', iteration=itr).
   
