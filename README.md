# rust-mlp
用rust训练mlp，以python接口形式调用
## 在自己的环境中使用
- 创建环境
```bash
conda create -n your_env_name python=3.11 # 必须>=3.9
conda activate your_env_name
```
- 下载wheel文件后
```bash
pip install Path/to/*.whl
```
- 在python环境中使用
```Python
import rust_mlp
rust_mlp.let_me_try()
```
预期结果如下：
```
Epoch 10 | Train Loss: 1.628042
Epoch 20 | Train Loss: 1.206229
Epoch 30 | Train Loss: 0.876774
Epoch 40 | Train Loss: 0.737402
Epoch 50 | Train Loss: 0.644500
Epoch 60 | Train Loss: 0.576381
Epoch 70 | Train Loss: 0.524079
Epoch 80 | Train Loss: 0.482977
Epoch 90 | Train Loss: 0.451547
Epoch 100 | Train Loss: 0.431264
Epoch 110 | Train Loss: 0.417604
Epoch 120 | Train Loss: 0.407959
Epoch 130 | Train Loss: 0.400804
Epoch 140 | Train Loss: 0.395103
Epoch 150 | Train Loss: 0.390289
Epoch 160 | Train Loss: 0.386080
Epoch 170 | Train Loss: 0.382233
Epoch 180 | Train Loss: 0.378714
Epoch 190 | Train Loss: 0.375436
Epoch 200 | Train Loss: 0.372300
Test Loss: 0.385307
```
