安装
====

基本安装
--------

.. code-block:: bash

   # 基本安装
   pip install torch-sla

   # 包含 cuDSS 支持 (CUDA 12+，GPU 推荐)
   pip install torch-sla[cuda]

   # 完整安装，包含所有依赖
   pip install torch-sla[all]

从源码安装
----------

.. code-block:: bash

   # 克隆仓库
   git clone https://github.com/walkerchi/torch-sla.git
   cd torch-sla

   # 开发模式安装
   pip install -e ".[dev]"

系统要求
--------

- Python >= 3.8
- PyTorch >= 1.10.0
- SciPy（CPU 推荐）
- CUDA Toolkit（GPU 后端需要）
- nvidia-cudss-cu12（可选，cuDSS 后端需要）

后端依赖
--------

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - 后端
     - 依赖
     - 安装方式
   * - ``scipy``
     - scipy
     - ``pip install scipy``
   * - ``cudss``
     - nvidia-cudss-cu12
     - ``pip install nvidia-cudss-cu12``
   * - ``cusolver``
     - CUDA Toolkit
     - 随 CUDA 安装
   * - ``pytorch``
     - torch
     - 已包含

验证安装
--------

.. code-block:: python

   import torch
   from torch_sla import SparseTensor, get_available_backends

   # 检查可用后端
   print("可用后端:", get_available_backends())

   # 快速测试
   A = SparseTensor.from_dense(torch.eye(3, dtype=torch.float64))
   b = torch.ones(3, dtype=torch.float64)
   x = A.solve(b)
   print("求解结果:", x)

