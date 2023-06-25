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
    constexpr size_t  buffer_size = 1 << 19;
    do
    {
        unp.reserve_buffer(buffer_size);
        in.read(unp.buffer(), buffer_size);
        unp.buffer_consumed(in.gcount());
        finished_parsing = unp.next(result); // may throw msgpack::parse_error
    } while(!finished_parsing && !in.fail());
    Tensile::Serialization::MessagePackInput min(result.get());
}
