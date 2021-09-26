from setuptools import setup
from Cython.Build import cythonize

setup(
    name='movey',
    ext_modules=cythonize(
      [
        "cell_list.pyx",
        "common.pyx",
        "run_opt.pyx"
      ],
      compiler_directives={'language_level' : "3"},
    ),
    zip_safe=False,
)
