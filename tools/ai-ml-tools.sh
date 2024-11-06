# Create a robust AI/ML environment
conda create -n ml-env python=3.11
conda activate ml-env

# Essential ML tools
pip install -U \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    tensorflow \
    scikit-learn \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    jupyter \
    jupyterlab \
    ipywidgets \
    plotly \
    opencv-python \
    transformers \
    datasets \
    accelerate \
    evaluate \
    gradio \
    wandb \
    pytorch-lightning \
    optuna

# VS Code extensions for ML
code --install-extension ms-python.python \
     --install-extension ms-toolsai.jupyter \
     --install-extension ms-toolsai.vscode-jupyter-cell-tags \
     --install-extension ms-toolsai.jupyter-renderers \
     --install-extension ms-toolsai.vscode-jupyter-slideshow
