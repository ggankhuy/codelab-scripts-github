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

#include <msgpack.hpp> // satisfied by: "apt install libmsgpack-dev -y"

// rocm hip/rocblas header

#include "hip/hip_runtime_api.h"
#include "rocblas.h"

// tensile library. 

#include <Tensile/Tensile.hpp>
#include <Tensile/msgpack/MessagePack.hpp>
#include <Tensile/msgpack/Loading.hpp>

#define USE_TENSILE_API_FULL 1
using namespace std;

void* context = nullptr;

namespace Tensile
{
    namespace Serialization
    {
        #if USE_TENSILE_API_FULL == 0
        Serialization::MessagePackInput createSubRef(msgpack::object otherObject)
        {
            return Tensile::Serialization::MessagePackInput (otherObject, context);
            //return Tensile::Serialization::MessagePackInput (otherObject, );
        }

        void objectToMapLike(msgpack::object & pObject, std::unordered_map<std::string, msgpack::object>& result) {
            for(uint32_t i = 0; i < pObject.via.map.size; i++) {
                auto& element = pObject.via.map.ptr[i];

                std::string key;
                switch(element.key.type)
                {
                case msgpack::type::object_type::STR:
                {
                element.key.convert(key);
                //std::cout << std::setw(100) << "  DBG object_type::STR: " << element.val << std::endl;
                std::cout << "  DBG object_type::STR, size(element.val):  " << sizeof(element.val) << std::endl;
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
                throw std::runtime_error("Unexpected map key type");
                }
            }
        }
        #endif

        int main() {
            std::string filename="/opt/rocm/lib/rocblas/library/TensileLibrary_gfx908.dat";
            msgpack::object_handle result;
            std::ifstream in(filename, std::ios::in | std::ios::binary);
            msgpack::unpacker unp;
            bool              finished_parsing;
            //constexpr size_t  buffer_size = 1 << 19;
            constexpr size_t  buffer_size = 1 << (19+8);

            std::unordered_map<std::string, msgpack::object> objectMap;
            std::unordered_set<std::string>          usedKeys;

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

            #if USE_TENSILE_API_FULL == 1
            // template is problematic

            //std::shared_ptr<MasterSolutionLibrary<MyProblem, MySolution>> rv;
            std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>> rv;

            //auto result = msgpack::unpack((const char*)data.data(), data.size());
            Tensile::Serialization::MessagePackInput min(result.get());

            Tensile::Serialization::PointerMappingTraits<Tensile::MasterContractionLibrary, \
            Tensile::Serialization::MessagePackInput>::mapping(min, rv);

            #else
            
            msgpack::object obj1=result.get();

            if(objectMap.empty())
            objectToMapLike(obj1, objectMap);
            
            auto iterator = objectMap.find("");

            if(iterator != objectMap.end())
            {
            auto&    value  = iterator->second;
            Tensile::Serialization::MessagePackInput  subRef = createSubRef(value);
            subRef.input(obj1);
            error.insert(error.end(), subRef.error.begin(), subRef.error.end());
            usedKeys.insert(key);
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
            addError(msg);
            #endif
            return 0;
        } // main        
    } // namespace serialization
} // namespace tensile
