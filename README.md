# GAN for Rain Generation

# Deep (Dilated) Convolutional Network for Deraining

## Installation

```bash
git clone https://github.com/KaneWANG0816/RGR.git
```

### Packages
```bash
pip install -r requirements.txt
```
### Environment:

Python 3.8

CUDA 11.3

### Pre-trained model
[Models](Models) contain pre-trained [Generator](Models/G_state_100.pt) and [PReNet](Models/net_epoch10.pth)

## Usage

<b>Default using GPU</b>

### Quick Test
```bash
python test_PReNet.py
```

### Or follow complete flow
1. [generateDataset.py](generateDataset.py) for generating [training](out/train) and [test](out/test) datasets for PReNet, from [test dataset of Rain100L](rain100L/test)
```bash
python generateDataset.py --out_path ./out/train --num 500
python generateDataset.py --out_path ./out/test --num 100
```
2. [PReNet](PReNet.py) trains with pairs of real Rain100L([rain](out/train/rain), [norain](out/train/norain))
```bash
python PReNet.py
```
4. Execute [test_PReNet.py](test_PReNet.py) to compare Real([rain](out/test/rain), [norain](out/test/norain)) and synthesised([rain_](out/test/rain_), [norain](out/test/norain))
```bash
python test_PReNet.py
```
4. Derain results are shown in [out/derain](out/derain)(rain100L) and [out/derain_](out/derain_)(synthesised rain)



### Try pre-trained derain model
Create a folder called 'out' with two subfolder called 'rain100L' and 'rain100H'<br>
where the derain result shows<br>

[derain.py](derain.py) for test on derain samples


