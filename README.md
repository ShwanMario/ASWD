# Augmented-Sliced-Wasserstein-Distances

## Python version

python==3.7.7

## Python packages
pytorch==1.4.0 

torchvision==0.5.0

cudatoolkit==10.1.243

cupy==7.40

numpy==1.18.1

pot==0.7.0

imageio=2.8.0

# Running the experiments
This code allows you to evaluate the performances of GANs trained with different variants of sliced-based Wasserstein distance on the CIFAR10 dataset. To train and evaluate the model, run the following commands.

```
python main.py  --model-type ASWD --dataset CIFAR --epochs 200 --num-projection 1000 --batch-size 512 --lr 0.0005
```
