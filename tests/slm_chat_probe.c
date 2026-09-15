/* Emit the production chat prefix so an independent HF template can verify it. */
#define main slm_backend_main
#include "../src/slm_run.c"
#undef main
int main(int argc,char **argv){
    if(argc!=5&&argc!=7)return 2;
    Model *m=load_model(argv[1],128);slm_tokenizer *tok=slm_tokenizer_load(argv[2]);if(!tok)return 2;
    Chat c={0};c.s=new_state(m);c.tok=tok;c.history=alloc((size_t)m->ctx,sizeof(int));c.scratch=alloc((size_t)m->ctx,sizeof(int));
    snprintf(c.system,sizeof(c.system),"%s",argv[3]);if(!chat_reset(&c))return 3;
    if(argc==7){
        int n=role_message(&c,"user",argv[5],1,c.scratch,m->ctx);
        if(n<0)return 3;
        int k=role_message(&c,"assistant",argv[6],1,c.scratch+n,m->ctx-n);
        if(k<0)return 3;
        append(&c,c.scratch,n+k,0);
    }
    int n=role_message(&c,"user",argv[4],1,c.scratch,m->ctx);
    if(n<0)return 3;
    int k=role_message(&c,"assistant","",0,c.scratch+n,m->ctx-n);
    if(k<0)return 3;
    append(&c,c.scratch,n+k,0);
    printf("[");for(int i=0;i<c.used;i++)printf("%s%d",i?",":"",c.history[i]);printf("]\n");
    free(c.history);free(c.scratch);free_state(c.s);free_model(m);slm_tokenizer_free(tok);return 0;
}
