API Reference
=============

This section provides the core API documentation for torch-sla.

.. raw:: html

   <ul class="feature-list">
     <li><span class="gradient-text">SparseTensor</span>: Core class for 2D sparse matrices with full PyTorch autograd support</li>
     <li><span class="gradient-text">SparseTensorList</span>: Container for batched sparse matrices with different layouts</li>
     <li><span class="gradient-text">DSparseTensor</span>: Distributed sparse tensor with domain decomposition and halo exchange</li>
   </ul>

----

SparseTensor
------------

The main class for working with sparse matrices.

.. autoclass:: torch_sla.SparseTensor
   :members:
   :undoc-members:
   :show-inheritance:

SparseTensorList
----------------

Container for multiple sparse matrices with different layouts.

.. autoclass:: torch_sla.SparseTensorList
   :members:
   :undoc-members:
   :show-inheritance:

DSparseTensor
-------------

Distributed sparse tensor with domain decomposition support.

.. autoclass:: torch_sla.DSparseTensor
   :members:
   :undoc-members:
   :show-inheritance:
