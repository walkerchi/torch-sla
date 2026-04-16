Installation
============

.. raw:: html

   <ul class="feature-list">
     <li><span class="gradient-text">PyPI</span>: <code>pip install torch-sla</code> — simplest installation</li>
     <li><span class="gradient-text">GitHub</span>: Clone and install for development</li>
     <li><span class="gradient-text">Optional backends</span>: cuDSS, Eigen for enhanced performance</li>
   </ul>

----

Using pip
---------

To install the latest release:

.. code-block:: bash

    pip install torch-sla

Or install from GitHub for the latest development version:

.. code-block:: bash

    pip install git+https://github.com/walkerchi/torch-sla.git

Optional Dependencies
---------------------

For additional backends and features:

.. code-block:: bash

    # With cuDSS support (requires CUDA 12+)
    pip install torch-sla[cuda]

    # Full installation with all optional dependencies
    pip install torch-sla[all]

    # For development
    pip install torch-sla[dev]

.. raw:: html

   <div class="recommendation-box">
     <h4><span class="gradient-text">cuDSS Now on PyPI!</span></h4>
     <p>CUDA backends use <code>nvmath-python</code> (for cuDSS) and <code>cupy-cuda12x</code> (for CuPy).
     Installing <code>torch-sla[cuda]</code> will automatically install them.</p>
   </div>

Backend Requirements
--------------------

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Backend
     - Installation
     - Notes
   * - ``scipy``
     - ``pip install scipy``
     - Default, always available
   * - ``pytorch``
     - Included with PyTorch
     - Native CG/BiCGStab solvers
   * - ``cupy``
     - ``pip install cupy-cuda12x``
     - GPU direct + iterative solvers via cupyx.scipy
   * - ``cudss``
     - ``pip install nvmath-python[cu12]``
     - Best for medium-scale GPU problems (10K-2M DOF)
