### Python wrapper for c++ implemented murty, gibbs, fuzzy, bbox overlap, esf, etc,.

------------------------------------
How to install this Python Wrapper
1) Clone FuzzyLite: `git clone https://github.com/fuzzylite/fuzzylite.git`  
   `cd fuzzylite`, `git checkout 5e8a1015e529ebba5a3d19430a05e51eef5f4a8b`  
   You can also configure FuzzyLite with it latest code but need to configure `CMakeLists.txt`  
2) Clone Pybind11: `git clone https://github.com/pybind/pybind11.git`
3) _Build FuzzyLite_
    1. Build your own FIS in Matlab, export to *.fis file, and copy to `tests` folder. You can also customize FIS in `build_engine()`, `fuzzy.hpp`. However, Matlab Fuzzy Logic Designer would be an easy tool to build and visualize your FIS. 
    2. Build fuzzylite first to get static library in `./fuzzylite/release/bin`: 
`cd fuzzylite` => `./build.sh` or  `build.bat` in Windows.
4) Move to the root folder and run: `python setup.py build develop`

------------------------------------
Notes: 
1) If you have a problem while building `fuzzylite` in Windows, e.g. _The C compiler is not able to compile a simple test program_, 
2) run this command on your terminal `vcvarsall.bat x64`. Specify your Windows Platform by `x86_amd64`, `x64`, etc.
3) Path example: `C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvarsall.bat x64`

Install Eigen for Windows (after the following steps, add include directory `C:\eigen-3.4.0` for example.)
1) Download Eigen 3.4.0 (NOT lower than this version) from it official website https://eigen.tuxfamily.org/ or [ZIP file here](https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip).
2) `mkdir build_dir`
3) `cd build_dir`
4) `cmake ../`
5) `make install`, this step does not require

Install Eigen for Linux
1) [install and use eigen3 on ubuntu 16.04](https://kezunlin.me/post/d97b21ee/) 
2) `sudo apt-get install libeigen3-dev` libeigen3-dev is installed install to `/usr/include/eigen3/` and `/usr/lib/cmake/eigen3`.
3) Thus, we must make a change in **CMakeLists.txt** `SET( EIGEN3_INCLUDE_DIR "/usr/local/include/eigen3" )` to `SET( EIGEN3_INCLUDE_DIR "/usr/include/eigen3/" )`.

Murty Implementations
1) [headers/murty.hpp] https://github.com/jonatanolofsson/mht/tree/master/murty 
2) [headers/MurtyMiller.hpp] https://github.com/jonatanolofsson/MurtyAlgorithm
3) https://github.com/fbaeuerlein/MurtyAlgorithm
4) https://github.com/motrom/fastmurty
5) https://github.com/gatagat/lap

### Contact
Linh Ma (linh.mavan@gm.gist.ac.kr), Machine Learning & Vision Laboratory, GIST, South Korea

### Citation
If you find this project useful in your research, please consider citing by:

```
@article{linh2024inffus,
      title={Visual Multi-Object Tracking with Re-Identification and Occlusion Handling using Labeled Random Finite Sets}, 
      author={Linh~Van~Ma and Tran~Thien~Dat~Nguyen and Changbeom~Shim and Du~Yong~Kim and Namkoo~Ha and Moongu~Jeon},
      journal={Pattern Recognition},
      volume = {156},
      year={2024},
      publisher={Elsevier}
}
```