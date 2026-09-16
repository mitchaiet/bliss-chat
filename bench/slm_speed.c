/* Fixed-token prefill/decode benchmark; pass 0 includes cold page faults. */
#define main slm_backend_main
#include "../src/slm_run.c"
#undef main
/* CPU time on POSIX; elapsed high-resolution time on Windows XP. */
static double seconds(void){
#ifdef _WIN32
    LARGE_INTEGER now,frequency;QueryPerformanceCounter(&now);QueryPerformanceFrequency(&frequency);
    return (double)now.QuadPart/(double)frequency.QuadPart;
#else
    return (double)clock()/CLOCKS_PER_SEC;
#endif
}
int main(int argc,char **argv){
    if(argc<2||argc>3)return 2;
    if(argc==3&&!strcmp(argv[2],"--float-activations"))float_activations=1;
    Model *m=load_model(argv[1],512);State *s=new_state(m);
    for(int pass=0;pass<3;pass++){
        reset_state(s);double start=seconds();
        for(int i=0;i<16;i++)forward(s,(i*317+19)%m->vocab,0);
        double prefill=seconds()-start;start=seconds();
        for(int i=0;i<16;i++)forward(s,(i*773+31)%m->vocab,1);
        double decode=seconds()-start;
        printf("{\"pass\":%d,\"prefill_tokens\":16,\"prefill_s\":%.6f,\"decode_tokens\":16,\"decode_s\":%.6f,\"decode_tok_s\":%.6f}\n",pass,prefill,decode,16/decode);fflush(stdout);
    }
    free_state(s);free_model(m);return 0;
}
