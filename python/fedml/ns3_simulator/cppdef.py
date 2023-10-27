from ns import ns

ns.cppyy.cppdef("""
        #ifndef common_cpp
        #define common_cpp

        using namespace ns3;

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

        EventImpl* make_offline_online_event(void(*func)(Ptr<Socket>), Ptr<Socket> socket)
        {
            return MakeEvent(func, socket);
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
        
        #endif
    """)