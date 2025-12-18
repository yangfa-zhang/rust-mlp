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
import json
with open('fetch_california_housing.json', encoding='utf-8') as f:
    data = json.load(f) 
X_train = data['X_train']
X_test  = data['X_test']
y_train = data['y_train']
y_test  = data['y_test']

input_dim = len(X_train[0])
model = rust_mlp.MLP(input_dim,lr = 0.01)
model.train(X_train,y_train, epochs = 20)
model.evaluate(X_test,y_test)
```
预期结果如下：
```
Loss: 4.4525466
Loss: 3.3261764
Loss: 2.539338
Loss: 2.008424
Loss: 1.6563666
Loss: 1.4215927
Loss: 1.2626717
Loss: 1.1582772
Loss: 1.0978976
Loss: 1.0723116
Loss: 1.0687258
Loss: 1.073826
Loss: 1.0768875
Loss: 1.0705063
Loss: 1.0518045
Loss: 1.0223864
Loss: 0.9862392
Loss: 0.9472292
Loss: 0.9076066
Loss: 0.86793756
Evaluation Loss: 0.82283765
```
