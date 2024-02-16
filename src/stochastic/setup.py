# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['stochastic',
 'stochastic.processes',
 'stochastic.processes.continuous',
 'stochastic.processes.diffusion',
 'stochastic.processes.discrete',
 'stochastic.processes.noise',
 'stochastic.utils']

package_data = \
{'': ['*']}

install_requires = \
['numpy>=1.19,<2.0', 'scipy>=1.8,<2.0']

setup_kwargs = {
    'name': 'stochastic',
    'version': '0.7.0',
    'description': 'Generate realizations of stochastic processes',
    'long_description': "stochastic\n==========\n\n|build| |rtd| |codecov| |pypi| |pyversions|\n\n.. |build| image:: https://github.com/crflynn/stochastic/actions/workflows/build.yml/badge.svg\n    :target: https://github.com/crflynn/stochastic/actions\n\n.. |rtd| image:: https://img.shields.io/readthedocs/stochastic.svg\n    :target: http://stochastic.readthedocs.io/en/latest/\n\n.. |codecov| image:: https://codecov.io/gh/crflynn/stochastic/branch/master/graphs/badge.svg\n    :target: https://codecov.io/gh/crflynn/stochastic\n\n.. |pypi| image:: https://img.shields.io/pypi/v/stochastic.svg\n    :target: https://pypi.python.org/pypi/stochastic\n\n.. |pyversions| image:: https://img.shields.io/pypi/pyversions/stochastic.svg\n    :target: https://pypi.python.org/pypi/stochastic\n\n\nA python package for generating realizations of stochastic processes.\n\nInstallation\n------------\n\nThe ``stochastic`` package is available on pypi and can be installed using pip\n\n.. code-block:: shell\n\n    pip install stochastic\n\nDependencies\n~~~~~~~~~~~~\n\nStochastic uses ``numpy`` for many calculations and ``scipy`` for sampling\nspecific random variables.\n\nProcesses\n---------\n\nThis package offers a number of common discrete-time, continuous-time, and\nnoise process objects for generating realizations of stochastic processes as\n``numpy`` arrays.\n\nThe diffusion processes are approximated using the Euler–Maruyama method.\n\nHere are the currently supported processes and their class references within\nthe package.\n\n* stochastic.processes\n\n    * continuous\n\n        * BesselProcess\n        * BrownianBridge\n        * BrownianExcursion\n        * BrownianMeander\n        * BrownianMotion\n        * CauchyProcess\n        * FractionalBrownianMotion\n        * GammaProcess\n        * GeometricBrownianMotion\n        * InverseGaussianProcess\n        * MixedPoissonProcess\n        * MultifractionalBrownianMotion\n        * PoissonProcess\n        * SquaredBesselProcess\n        * VarianceGammaProcess\n        * WienerProcess\n\n    * diffusion\n\n        * DiffusionProcess (generalized)\n        * ConstantElasticityVarianceProcess\n        * CoxIngersollRossProcess\n        * ExtendedVasicekProcess\n        * OrnsteinUhlenbeckProcess\n        * VasicekProcess\n\n    * discrete\n\n        * BernoulliProcess\n        * ChineseRestaurantProcess\n        * DirichletProcess\n        * MarkovChain\n        * MoranProcess\n        * RandomWalk\n\n    * noise\n\n        * BlueNoise\n        * BrownianNoise\n        * ColoredNoise\n        * PinkNoise\n        * RedNoise\n        * VioletNoise\n        * WhiteNoise\n        * FractionalGaussianNoise\n        * GaussianNoise\n\nUsage patterns\n--------------\n\nSampling\n~~~~~~~~\n\nTo use ``stochastic``, import the process you want and instantiate with the\nrequired parameters. Every process class has a ``sample`` method for generating\nrealizations. The ``sample`` methods accept a parameter ``n`` for the quantity\nof steps in the realization, but others (Poisson, for instance) may take\nadditional parameters. Parameters can be accessed as attributes of the\ninstance.\n\n.. code-block:: python\n\n    from stochastic.processes.discrete import BernoulliProcess\n\n\n    bp = BernoulliProcess(p=0.6)\n    s = bp.sample(16)\n    success_probability = bp.p\n\n\nContinuous processes provide a default parameter, ``t``, which indicates the\nmaximum time of the process realizations. The default value is 1. The sample\nmethod will generate ``n`` equally spaced increments on the\ninterval ``[0, t]``.\n\nSampling at specific times\n~~~~~~~~~~~~~~~~~~~~~~~~~~\n\nSome continuous processes also provide a ``sample_at()`` method, in which a\nsequence of time values can be passed at which the object will generate a\nrealization. This method ignores the parameter, ``t``, specified on\ninstantiation.\n\n\n.. code-block:: python\n\n    from stochastic.processes.continuous import BrownianMotion\n\n\n    bm = BrownianMotion(drift=1, scale=1, t=1)\n    times = [0, 3, 10, 11, 11.2, 20]\n    s = bm.sample_at(times)\n\nSample times\n~~~~~~~~~~~~\n\nContinuous processes also provide a method ``times()`` which generates the time\nvalues (using ``numpy.linspace``) corresponding to a realization of ``n``\nsteps. This is particularly useful for plotting your samples.\n\n\n.. code-block:: python\n\n    import matplotlib.pyplot as plt\n    from stochastic.processes.continuous import FractionalBrownianMotion\n\n\n    fbm = FractionalBrownianMotion(hurst=0.7, t=1)\n    s = fbm.sample(32)\n    times = fbm.times(32)\n\n    plt.plot(times, s)\n    plt.show()\n\n\nSpecifying an algorithm\n~~~~~~~~~~~~~~~~~~~~~~~\n\nSome processes provide an optional parameter ``algorithm``, in which one can\nspecify which algorithm to use to generate the realization using the\n``sample()`` or ``sample_at()`` methods. See the documentation for\nprocess-specific implementations.\n\n\n.. code-block:: python\n\n    from stochastic.processes.noise import FractionalGaussianNoise\n\n\n    fgn = FractionalGaussianNoise(hurst=0.6, t=1)\n    s = fgn.sample(32, algorithm='hosking')\n",
    'author': 'Flynn',
    'author_email': 'crf204@gmail.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://github.com/crflynn/stochastic',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8,<3.11',
}


setup(**setup_kwargs)
