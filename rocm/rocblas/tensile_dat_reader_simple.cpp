/*
msgpack tutorial:
https://github.com/msgpack/msgpack-c/wiki/v2_0_cpp_unpacker
code in this example apparently uses unpack option 2: msgpack controls buffer (option 1. client 
controls buffer is not seem to be the case here).
See under "msgpack controls a buffer"
*/

// std c++ library. Note msgpack.hpp!

#include <iostream> 
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <iterator>
#include <vector>
#include <chrono>
#include <thread>

#include <msgpack.hpp> // satisfied by: "apt install libmsgpack-dev -y"

// rocm hip/rocblas header

#include "hip/hip_runtime_api.h"
#include "rocblas.h"

// tensile library. 

#include <Tensile/Tensile.hpp>
#include <Tensile/msgpack/MessagePack.hpp>
#include <Tensile/msgpack/Loading.hpp>

using namespace std;
namespace Tensile {
    namespace Serialization {
        
        /*
        MessagePackInput createSubRef(msgpack::object otherObject)
        {
            return MessagePackInput(otherObject, context);
        }*/

        void objectToMap(msgpack::object & pObject, std::unordered_map<std::string, msgpack::object>& result, int level = 0) {
            cout << "objectToMap entered: level=" << level << endl;
            if (level >= 5) {
                cout << "Recursive limit reached." << endl;
                return; 
            }

            for(uint32_t i = 0; i < pObject.via.map.size; i++) {
                cout << "i: " << i << endl;
                auto& element = pObject.via.map.ptr[i];

                std::string key;
                switch(element.key.type)
                {
                case msgpack::type::object_type::STR:
                {
                    element.key.convert(key);
                    //std::cout << std::setw(100) << "  DBG object_type::STR: " << element.val << std::endl;
                    std::cout << "  DBG object_type::STR, size(element.val):  " << sizeof(element.val) << std::endl;
                    level += 1;
                    break;
                }
                case msgpack::type::object_type::POSITIVE_INTEGER:
                {
                    auto iKey = element.key.as<uint32_t>();
                    key       = std::to_string(iKey);
                    std::cout << std::setw(100) << "  DBG: key set to: " << key << std::endl;
                    break;
                }
                default:
                    cout << "element.key.type: " << element.key.type << endl;
                    throw std::runtime_error("Unexpected map key type");
                }
                result[key] = std::move(element.val);
            }
        }

    }
}

using namespace Tensile::Serialization;

        int main() {
            std::string filename="/opt/rocm/lib/rocblas/library/TensileLibrary_gfx908.dat";
            msgpack::object_handle result;
            std::ifstream in(filename, std::ios::in | std::ios::binary);
            msgpack::unpacker unp;
            bool              finished_parsing;
            //constexpr size_t  buffer_size = 1 << 19;
            constexpr size_t  buffer_size = 1 << (19+8);

            std::string key = "version";
            int counter = 0;
            int recur_level = 1;
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

            std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>> rv;
            //std::shared_ptr<MasterSolutionLibrary<Tensile::ContractionProblem>> library;

            //MessagePackInput min(result.get());
            /*PointerMappingTraits<Tensile::MasterContractionLibrary,
                                                MessagePackInput>::mapping(min, rv);
            */

            msgpack::object obj=result.get();

            std::unordered_map<std::string, msgpack::object> objectMap;
            std::unordered_set<std::string>                  usedKeys;
            void* context = nullptr;

            objectToMap(obj, objectMap, recur_level);

            for (const auto& [key, value] : objectMap) {
                //cout << "objectMap: " << key << ", " << value  << endl;
                cout << "objectMap: KEY: " << key << ", value: " << typeid(value).name() <<  endl;
                //cout << "objectMap: ";
            }
            auto iterator = objectMap.find(key);
            
            if(iterator != objectMap.end())
            {
                auto&    value  = iterator->second;
                //cout << "value: " << value << endl;
                MessagePackInput subRef = MessagePackInput(value, context);
                //MessagePackInput subRef = createSubRef(value);
                //subRef.input(obj);
                //error.insert(error.end(), subRef.error.begin(), subRef.error.end());
                //if(Tensile::Debug::Instance().printDataInit())
                //    usedKeys.insert(key);
            }
            else
            {
                std::string msg = "Unknown key ";
                msg += key;
                msg += " (keys: ";
                bool first = true;
                for(auto const& pair : objectMap)
                {
                    if(!first)
                        msg += ", ";
                    msg += pair.first;
                    first = false;
                }
                msg += ")";
                //addError(msg);
                cout << "Err: msg: " << msg << endl;
            }
            return 0;
        }
