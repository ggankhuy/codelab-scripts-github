/*
msgpack tutorial:
https://github.com/msgpack/msgpack-c/wiki/v2_0_cpp_unpacker
code in this example apparently uses unpack option 2: msgpack controls buffer (option 1. client 
controls buffer is not seem to be the case here).
See under "msgpack controls a buffer"
*/
#include <msgpack.hpp>
#include <iostream>
#include <fstream>
#include "hip/hip_runtime_api.h"
#include "rocblas.h"
#include <mutex>
#include <shared_mutex>
#include <iterator>
#include <vector>

#include <Tensile/Tensile.hpp>
#include <Tensile/msgpack/MessagePack.hpp>
#include <Tensile/msgpack/Loading.hpp>

using namespace std;

int main() {
    std::string filename="/opt/rocm/lib/rocblas/library/TensileLibrary_gfx908.dat";
    msgpack::object_handle result;
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    msgpack::unpacker unp;
    bool              finished_parsing;
    //constexpr size_t  buffer_size = 1 << 19;
    constexpr size_t  buffer_size = 1 << (19+8);

    int counter = 0;
    do
    {
        unp.reserve_buffer(buffer_size);
        in.read(unp.buffer(), buffer_size);
        unp.buffer_consumed(in.gcount());
        finished_parsing = unp.next(result); // may throw msgpack::parse_error
        counter += 1;
        cout << counter << ".." << endl;
        cout << "finished parsing? " << finished_parsing << endl;
        cout << "in.gcount: " << in.gcount() << endl;
    } while(!finished_parsing && !in.fail());
    //Tensile::Serialization::MessagePackInput min(result.get());
    msgpack::object obj1=result.get();
    for(uint32_t i = 0; i < obj1.via.map.size; i++) {
        auto& element = obj1.via.map.ptr[i];

                std::string key;
                switch(element.key.type)
                {
                case msgpack::type::object_type::STR:
                {
                    element.key.convert(key);
                    std::cout << "  DBG object_type::STR: " << element.val << std::endl;
                    break;
                }
                case msgpack::type::object_type::POSITIVE_INTEGER:
                {
                    auto iKey = element.key.as<uint32_t>();
                    key       = std::to_string(iKey);
                    std::cout << "  DBG: key set to: " << key << std::endl;
                    break;
                }
                default:
                    throw std::runtime_error("Unexpected map key type");
                }
        
    }
}
