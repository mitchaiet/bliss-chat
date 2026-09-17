/* Optional, disposable system-prefix cache. Included after the Chat helpers.
 * The fixed header identifies the exact model, executable and state layout.
 * Every size comes from the validated model, never from untrusted file data. */
#ifndef SLM_PREFIX_DISK_H
#define SLM_PREFIX_DISK_H
static size_t prefix_elements(const Chat *c,int n){
    Model *m=c->s->m;
    return checked_size(2ull*m->na*(unsigned)n*m->nk*m->hd+(uint64_t)m->nc*m->d*m->conv_width);
}
static void prefix_put32(unsigned char *p,uint32_t v){for(int i=0;i<4;i++)p[i]=(unsigned char)(v>>(i*8));}
static void prefix_header(const Chat *c,int n,unsigned char header[128]){
    memset(header,0,128);memcpy(header,"SLMPFX1",7);prefix_put32(header+8,1);
    prefix_put32(header+12,(uint32_t)n);prefix_put32(header+16,(uint32_t)c->s->m->ctx);
    prefix_put32(header+20,(uint32_t)prefix_elements(c,n));
    prefix_put32(header+24,(uint32_t)float_activations);
    /* Native float payloads are only reusable on the same executable/platform. */
    prefix_put32(header+28,0x01020304);memcpy(header+32,c->prefix_identity,64);
}
static void prefix_disk_init(Chat *c,const char *path,const char *executable){
    if(!path||!*path||strlen(path)>900)return;
    FILE *f=fopen(executable,"rb");if(!f)return;
    SlmSha256 hash;slm_sha256_init(&hash);unsigned char buffer[16384];size_t n;
    while((n=fread(buffer,1,sizeof(buffer),f))>0)slm_sha256_update(&hash,buffer,n);
    int okay=!ferror(f);if(fclose(f))okay=0;if(!okay)return;
    slm_sha256_final(&hash,c->prefix_identity+32);
    slm_sha256_init(&hash);slm_sha256_update(&hash,c->s->m->map,c->s->m->mapsize);
    slm_sha256_final(&hash,c->prefix_identity);c->prefix_path=path;
}
static void prefix_payload_hash(const Chat *c,unsigned char digest[32]){
    SlmSha256 hash;slm_sha256_init(&hash);
    slm_sha256_update(&hash,c->prefix_ids,(size_t)c->prefix_tokens*sizeof(int));
    slm_sha256_update(&hash,c->prefix_state,prefix_elements(c,c->prefix_tokens)*sizeof(float));
    slm_sha256_final(&hash,digest);
}
static int prefix_disk_load(Chat *c,const int *ids,int n){
    if(!c->prefix_path||n<1||n>c->s->m->ctx/2)return 0;
    FILE *f=fopen(c->prefix_path,"rb");if(!f)return 0;
    unsigned char actual[128],expected[128],digest[32];prefix_header(c,n,expected);
    size_t count=prefix_elements(c,n),idbytes=(size_t)n*sizeof(int);
    if(count>((size_t)LONG_MAX-128-idbytes)/sizeof(float)){fclose(f);return 0;}
    size_t bytes=count*sizeof(float);
    int okay=fread(actual,1,128,f)==128&&!memcmp(actual,expected,96);
    if(okay)okay=!fseek(f,0,SEEK_END)&&ftell(f)==(long)(128+idbytes+bytes)&&!fseek(f,128,SEEK_SET);
    if(okay){
        c->prefix_ids=malloc(idbytes);c->prefix_state=malloc(bytes?bytes:1);
        c->prefix_tokens=n;okay=c->prefix_ids&&c->prefix_state;
        if(okay)okay=fread(c->prefix_ids,1,idbytes,f)==idbytes&&fread(c->prefix_state,1,bytes,f)==bytes&&!memcmp(c->prefix_ids,ids,idbytes);
        if(okay){prefix_payload_hash(c,digest);okay=!memcmp(digest,actual+96,32);}
    }
    if(fclose(f))okay=0;
    if(!okay){free_prefix(c);return 0;}
    c->prefix=n;c->used=n;memcpy(c->history,c->prefix_ids,idbytes);transfer_prefix(c,1);
    fprintf(stderr,"[slm] prefix cache hit: %d tokens\n",n);return 1;
}
static void prefix_disk_save(const Chat *c){
    if(!c->prefix_path||prefix_elements(c,c->prefix_tokens)>((size_t)LONG_MAX-128-(size_t)c->prefix_tokens*sizeof(int))/sizeof(float))return;
    char temporary[960];unsigned long pid;
#ifdef _WIN32
    pid=(unsigned long)GetCurrentProcessId();
#else
    pid=(unsigned long)getpid();
#endif
    snprintf(temporary,sizeof(temporary),"%s.tmp.%lu",c->prefix_path,pid);
#ifdef _WIN32
    int fd=_open(temporary,_O_WRONLY|_O_CREAT|_O_EXCL|_O_BINARY,_S_IREAD|_S_IWRITE);
    FILE *f=fd<0?NULL:_fdopen(fd,"wb");
#else
    int fd=open(temporary,O_WRONLY|O_CREAT|O_EXCL,0600);
    FILE *f=fd<0?NULL:fdopen(fd,"wb");
#endif
    if(!f){if(fd>=0){
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
        remove(temporary);}return;}
    unsigned char header[128];prefix_header(c,c->prefix_tokens,header);prefix_payload_hash(c,header+96);
    size_t bytes=prefix_elements(c,c->prefix_tokens)*sizeof(float),idbytes=(size_t)c->prefix_tokens*sizeof(int);
    int okay=fwrite(header,1,128,f)==128&&fwrite(c->prefix_ids,1,idbytes,f)==idbytes&&fwrite(c->prefix_state,1,bytes,f)==bytes;
    if(fclose(f))okay=0;
#ifdef _WIN32
    if(okay)okay=MoveFileExA(temporary,c->prefix_path,MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)!=0;
#else
    if(okay)okay=rename(temporary,c->prefix_path)==0;
#endif
    if(!okay)remove(temporary);
    else fprintf(stderr,"[slm] prefix cache saved: %d tokens\n",c->prefix_tokens);
}
#endif
