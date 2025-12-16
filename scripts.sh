mkdir rust-mlp
cd rust-mlp
conda activate fafa
pip install maturin
maturin init 
maturin develop --release 
maturin build --release