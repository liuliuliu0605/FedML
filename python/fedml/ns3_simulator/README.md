<<<<<<< HEAD
# Prerequisites
```shell
sudo apt install -y openmpi-bin openmpi-common openmpi-doc libopenmpi-dev cmake
pip install cppyy==2.4.2
```

# Download and install ns3.40
```shell
wget https://www.nsnam.org/releases/ns-allinone-3.40.tar.bz2
tar xfj ns-allinone-3.40.tar.bz2
cd ns-allinone-3.40/ns-3.40
./ns3 distclean
./ns3 configure  --build-profile=optimized --enable-examples --enable-tests --enable-python-bindings --enable-mpi
./ns3
./ns3 show profile
./ns3 show config
```

# Set environment variables
=======
# install ns3
https://www.nsnam.org/docs/release/3.35/tutorial/ns-3-tutorial.pdf

# ns3 env
```shell
export NS3_HOME=/home/ubuntu/Software/ns-allinone-3.40/ns-3.40
export PYTHONPATH=$PYTHONPATH:$NS3_HOME/build/bindings/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NS3_HOME/build/lib
```

# Test
```shell
./test.py
./ns3 run simple-distributed --command-template="mpiexec -np 2 %s"
```

# Materials
https://www.nsnam.org/docs/tutorial/html/getting-started.html
https://gitlab.com/nsnam/ns-3-dev/-/tree/ns-3.40?ref_type=tags
https://www.nsnam.org/docs/manual/html/python.html
https://gitlab.com/nsnam/ns-3-dev/-/blob/master/doc/installation/source/quick-start.rst
https://www.nsnam.org/docs/release/3.40/tutorial/ns-3-tutorial.pdf
=======
# build
sudo apt install -y cmake mpich
pip install cppyy

./ns3 clean
./ns3 configure  --build-profile=optimized --enable-examples --enable-tests --enable-python-bindings --enable-mpi
./ns3 build
./ns3 show profile

# others
https://www.nsnam.org/wiki/Installation#Installation
https://www.nsnam.org/docs/tutorial/html/getting-started.html
https://gitlab.com/nsnam/ns-3-dev/-/tree/ns-3.40?ref_type=tags
https://www.nsnam.org/docs/manual/html/python.html
>>>>>>> 48a705599a9b85734f1609e62e7982ecf9554334
