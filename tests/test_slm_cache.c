/* A hybrid reset must discard convolution history as well as rewind attention. */
#define main slm_backend_main
#include "../src/slm_run.c"
#undef main
int main(int argc,char **argv){
    if(argc!=2)return 2;
    Model *m=load_model(argv[1],128);State *used=new_state(m),*fresh=new_state(m);
    for(int i=0;i<67;i++)forward(used,(i*317+19)%m->vocab,0);
    reset_state(used);
    for(int i=0;i<43;i++){
        int token=(i*773+31)%m->vocab;
        forward(used,token,1);forward(fresh,token,1);
        if(memcmp(used->logits,fresh->logits,(size_t)m->vocab*sizeof(float))){fprintf(stderr,"reset changed logits at position%d\n",i);return 1;}
    }
    printf("43 reset/replay positions are bit-identical to a fresh state (%d conv layers).\n",m->nc);
    free_state(used);free_state(fresh);free_model(m);return 0;
}
