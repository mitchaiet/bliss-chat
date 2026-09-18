/* Print the SHA-256 of each file argument, one per line: "<hex>  <path>".
 * Built for Windows XP with the same toolchain as the backend. */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "../src/slm_sha256.h"
int main(int argc,char **argv){
    int bad=0;
    for(int i=1;i<argc;i++){
        FILE *f=fopen(argv[i],"rb");
        if(!f){printf("ERROR  %s\n",argv[i]);bad=1;continue;}
        SlmSha256 h;slm_sha256_init(&h);unsigned char buf[65536];size_t n;
        while((n=fread(buf,1,sizeof(buf),f))>0)slm_sha256_update(&h,buf,n);
        if(ferror(f)){printf("ERROR  %s\n",argv[i]);bad=1;fclose(f);continue;}
        fclose(f);unsigned char d[32];slm_sha256_final(&h,d);
        for(int j=0;j<32;j++)printf("%02x",d[j]);printf("  %s\n",argv[i]);
    }
    return bad;
}
