#include <stdio.h>
#ifdef OPT
 #if OPT == 1
  N=1000;
 #elif OPT == 2
  N=2000;
 #endif
#endif
int main() {
    printf("main...\n");
    printf("N: %u.\n", N);
}
