#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

int main() {
    std::ifstream myfile;
    myfile.open("1-hipMalloc.out", std::ios::binary);
    int length = 64;

    char * buffer;
    buffer = new char [length];
    
    myfile.read(buffer, length);

    for (int i = 0 ; i < length; i+=16) { 
        for (int j = 0 ; j < 16 ; j++) {
            std::cout << std::hex << std::setfill('0') << std::setw(2) << (uint)buffer[i+j] << " ";
        }
        std::cout << std::endl << "buffer: " << std::endl << &buffer[i] << std::endl;
        
    }

    myfile.close();

    
    
}
