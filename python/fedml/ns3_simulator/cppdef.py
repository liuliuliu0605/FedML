from ns import ns

import mpi4py

ns.cppyy.add_include_path(mpi4py.get_include())
ns.cppyy.cppdef("""
        #ifndef common_cpp
        #define common_cpp
        
        #include <mpi4py/mpi4py.h>
        using namespace ns3;
            
        MPI_Comm * Convert2MPIComm(PyObject* py_src){
            if (!PyMPIComm_Get){
                if (import_mpi4py() < 0)
                  {
                  return NULL;
                  }
            }
            if (!PyObject_TypeCheck(py_src, &PyMPIComm_Type)){
                return NULL;
            }
            MPI_Comm *mpiComm = PyMPIComm_Get(py_src);
            return mpiComm;
        }
        
        Callback<void, Ptr<Socket>, unsigned int>
        make_write_callback(void(*func)(Ptr<Socket>, unsigned int))
        {
            return MakeCallback(func);
        }

        Callback<void, Ptr<Socket>>
        make_connection_succeeded_callback(void(*func)(Ptr<Socket>))
        {
            return MakeCallback(func);
        }

        Callback<void, Ptr<Socket>>
        make_connection_failed_callback(void(*func)(Ptr<Socket>))
        {
            return MakeCallback(func);
        }
        
        Callback<bool, Ptr<Socket>, const Address&>
        make_connection_request(bool(*func)(Ptr<Socket>, const Address&))
        {
            return MakeCallback(func);
        }
        
        Callback<void, Ptr<Socket>, const Address&>
        make_new_connection_created(void(*func)(Ptr<Socket>, const Address&))
        {
            return MakeCallback(func);
        }
        
        Callback<void, Ptr<Socket>>
        make_rcv_wrapper(void(*func)(Ptr<Socket>))
        {
            return MakeCallback(func);
        }
        
        Callback<void, Ptr<Socket>>
        make_error_close_callback(void(*func)(Ptr<Socket>))
        {
            return MakeCallback(func);
        }
        
        Callback<void, Ptr<Socket>>
        make_normal_close_callback(void(*func)(Ptr<Socket>))
        {
            return MakeCallback(func);
        }
        
        EventImpl* make_offline_online_event(void(*func)(Ptr<Socket>), Ptr<Socket> socket)
        {
            return MakeEvent(func, socket);
        }
        
        EventImpl* make_add_message_event(void(*func)(Ptr<Socket>, char*), Ptr<Socket> socket, char* message)
        {
            return MakeEvent(func, socket, message);
        }
        
        #endif
    """)