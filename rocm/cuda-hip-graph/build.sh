for i in 1 2 3; do sudo hipcc ex-3.cpp ; sudo rocprof --hip-trace ./a.out ; sudo mkdir 3 ; sudo cp results.json 3/ ; done
