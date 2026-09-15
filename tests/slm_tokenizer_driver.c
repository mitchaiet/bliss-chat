/* Persistent, sanitizer-friendly tokenizer test driver; no model is loaded. */
#include "slm_tokenizer.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if(argc!=3)return 3;
    slm_tokenizer *tok=slm_tokenizer_load(argv[1]);
    if(!tok)return 2;
    FILE *f=fopen(argv[2],"rb");if(!f){slm_tokenizer_free(tok);return 3;}
    printf("{\"vocab\":%d}\n",slm_tokenizer_vocab(tok));
    uint32_t header[4];
    while(fread(header,sizeof(uint32_t),4,f)==4){
        int op=(int)header[0],parameter=(int32_t)header[1],capacity=(int32_t)header[2];
        uint32_t length=header[3];
        assert(length<=1048576&&capacity<=65536);
        char *text=malloc((size_t)length+1);assert(text);
        assert(fread(text,1,length,f)==length);text[length]=0;
        if(op==0){
            int available=capacity>0?capacity:0;
            int *ids=malloc(((size_t)available+2)*sizeof(int));assert(ids);
            ids[0]=ids[available+1]=123456789;
            int n=slm_encode(tok,text,parameter,ids+1,capacity);
            assert(ids[0]==123456789&&ids[available+1]==123456789);
            assert(n==-1||(n>=0&&n<=available));
            printf("{\"count\":%d,\"ids\":[",n);
            for(int i=0;i<n;i++)printf("%s%d",i?",":"",ids[i+1]);
            printf("]}\n");free(ids);
        }else if(op==1){
            int n=-1,special=-1;
            const unsigned char *bytes=slm_token_bytes(tok,parameter,&n,&special);
            printf("{\"valid\":%s,\"length\":%d,\"special\":%d,\"hex\":\"",bytes?"true":"false",n,special);
            if(bytes)for(int i=0;i<n;i++)printf("%02x",bytes[i]);
            printf("\"}\n");
        }else assert(0);
        free(text);
    }
    assert(feof(f));fclose(f);slm_tokenizer_free(tok);return 0;
}
