# install ns3
https://www.nsnam.org/docs/release/3.35/tutorial/ns-3-tutorial.pdf

# ns3 env
```shell
export NS3_HOME=/home/ubuntu/Software/ns-3-dev
export PYTHONPATH=$PYTHONPATH:$NS3_HOME/build/bindings/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NS3_HOME/build/lib
```

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