from distutils.core import setup, Extension

module = Extension("cgal_Segmentation_Module",
                   sources = ["neuron_cgal_segmentation.cpp"],
                   extra_compile_args=['-v','-std=c++0x'],
                   extra_link_args=['-L /usr/include/','-lCGAL','-lgmp','-std=c++0x'])

setup(name="CGAL_Segmentation",
      version = "1.0",
      description = "This is a package for cgal_Segmentation_Module",
      ext_modules = [module])
