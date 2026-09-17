#include "../src/slm_sha256.h"
#include <stdio.h>
#include <assert.h>
static void finish(SlmSha256 *s,const char *expected){
    unsigned char digest[32];char text[65];slm_sha256_final(s,digest);
    for(int i=0;i<32;i++)sprintf(text+i*2,"%02x",digest[i]);
    assert(!strcmp(text,expected));
}
int main(void){
    SlmSha256 s;slm_sha256_init(&s);
    finish(&s,"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    slm_sha256_init(&s);slm_sha256_update(&s,"a",1);slm_sha256_update(&s,"bc",2);
    finish(&s,"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    const char *long_vector="abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    slm_sha256_init(&s);slm_sha256_update(&s,long_vector,strlen(long_vector));
    finish(&s,"248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    char chunk[1000];memset(chunk,'a',sizeof(chunk));slm_sha256_init(&s);
    for(int i=0;i<1000;i++)slm_sha256_update(&s,chunk,sizeof(chunk));
    finish(&s,"cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    puts("SHA-256: empty, abc, 56-byte padding boundary and million-a vectors pass.");return 0;
}
