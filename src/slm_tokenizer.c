/* SmolLM2/LFM2.5 tokenizers: versioned byte-level regex and ranked BPE.
 * The exporter stores decoded bytes and Unicode class ranges; no ICU/regex DLL is needed.
 * Algorithm follows Hugging Face tokenizers (Apache-2.0), with an independent C implementation.
 */
#include "slm_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
typedef struct { unsigned char *bytes; uint32_t len, special; } Token;
typedef struct { uint32_t left,right,out,rank; int used; } Merge;
typedef struct { uint32_t lo,hi,flags; } Range;
struct slm_tokenizer { int version,vocab,hashsize,nranges,nadded; int *added; Token *tokens; Merge *merges; Range *ranges; int byteids[256]; };
static int read32(FILE *f,uint32_t *p) { return fread(p,4,1,f)==1; }
static uint32_t pairhash(uint32_t a,uint32_t b) { return (a*0x9e3779b1u)^(b*0x85ebca6bu); }
void slm_tokenizer_free(slm_tokenizer *t) { if(!t)return; if(t->tokens)for(int i=0;i<t->vocab;i++)free(t->tokens[i].bytes); free(t->tokens);free(t->merges);free(t->ranges);free(t->added);free(t); }
slm_tokenizer *slm_tokenizer_load(const char *path) {
    FILE *f=fopen(path,"rb"); char magic[8]; uint32_t version,nv,nm,nr;
    slm_tokenizer *t=NULL;
    if(!f)return NULL;
    if(fread(magic,1,8,f)!=8||memcmp(magic,"SLMTOK1\0",8)||!read32(f,&version)||!read32(f,&nv)||!read32(f,&nm)||!read32(f,&nr)||(version!=1&&version!=2)||nv<256||nv>1000000||nm>2000000||nr>20000)goto fail;
    t=calloc(1,sizeof(*t));if(!t)goto fail;t->version=(int)version;t->vocab=(int)nv;t->nranges=(int)nr;
    t->tokens=calloc(nv,sizeof(Token));t->ranges=calloc(nr,sizeof(Range));
    t->hashsize=1;while(t->hashsize<(int)nm*2)t->hashsize*=2;
    t->merges=calloc((size_t)t->hashsize,sizeof(Merge));
    t->added=malloc((size_t)nv*sizeof(int));
    if(!t->tokens||!t->ranges||!t->merges||!t->added)goto fail;
    for(int b=0;b<256;b++)t->byteids[b]=-1;
    for(uint32_t i=0;i<nv;i++) {
        Token *p=t->tokens+i;
        if(!read32(f,&p->len)||!read32(f,&p->special)||p->len>1048576||p->special>(version==1?1u:3u)||(!p->len&&(p->special&2)))goto fail;
        p->bytes=malloc((size_t)p->len+1);if(!p->bytes)goto fail;
        if(fread(p->bytes,1,p->len,f)!=p->len)goto fail;
        p->bytes[p->len]=0;
        if(p->len==1&&!p->special)t->byteids[p->bytes[0]]=(int)i;
        if(p->len&&(p->special&(version==1?1u:2u)))t->added[t->nadded++]=(int)i;
    }
    for(uint32_t i=0;i<nm;i++) {
        uint32_t a,b,c;if(!read32(f,&a)||!read32(f,&b)||!read32(f,&c)||a>=nv||b>=nv||c>=nv)goto fail;
        uint32_t h=pairhash(a,b)&(uint32_t)(t->hashsize-1);
        while(t->merges[h].used){if(t->merges[h].left==a&&t->merges[h].right==b)goto fail;h=(h+1)&(uint32_t)(t->hashsize-1);}
        t->merges[h]=(Merge){a,b,c,i,1};
    }
    for(uint32_t i=0;i<nr;i++)if(!read32(f,&t->ranges[i].lo)||!read32(f,&t->ranges[i].hi)||!read32(f,&t->ranges[i].flags)||t->ranges[i].lo>t->ranges[i].hi||t->ranges[i].hi>0x10ffff||t->ranges[i].flags>7||(i&&t->ranges[i].lo<=t->ranges[i-1].hi))goto fail;
    if(fgetc(f)!=EOF)goto fail;
    fclose(f);return t;
fail: fclose(f);slm_tokenizer_free(t);return NULL;
}
int slm_tokenizer_vocab(slm_tokenizer *t){return t?t->vocab:0;}
const unsigned char *slm_token_bytes(slm_tokenizer *t,int id,int *length,int *special) {
    if(!t||id<0||id>=t->vocab)return NULL;
    *length=(int)t->tokens[id].len;*special=(int)(t->tokens[id].special&1);return t->tokens[id].bytes;
}
static int properties(slm_tokenizer *t,uint32_t c) {
    int lo=0,hi=t->nranges-1;
    while(lo<=hi){int m=lo+(hi-lo)/2;Range *r=t->ranges+m;if(c<r->lo)hi=m-1;else if(c>r->hi)lo=m+1;else return(int)r->flags;}return 0;
}
static int utf8(const unsigned char *p,int n,uint32_t *cp) {
    unsigned c=p[0];int k=c<128?1:c>=0xc2&&c<=0xdf?2:c>=0xe0&&c<=0xef?3:c>=0xf0&&c<=0xf4?4:0;
    if(!k||n<k){*cp=0xfffd;return 1;}uint32_t x=c&((1u<<(7-k))-1u);if(k==1)x=c;
    for(int j=1;j<k;j++){if((p[j]&0xc0)!=0x80){*cp=0xfffd;return 1;}x=(x<<6)|(p[j]&63);}
    if((k==2&&x<128)||(k==3&&x<2048)||(k==4&&x<65536)||x>0x10ffff||(x>=0xd800&&x<=0xdfff)){*cp=0xfffd;return 1;}
    *cp=x;return k;
}
static Merge *lookup(slm_tokenizer *t,int a,int b) {
    unsigned h=pairhash((uint32_t)a,(uint32_t)b)&(unsigned)(t->hashsize-1);
    while(t->merges[h].used){Merge *m=t->merges+h;if(m->left==(uint32_t)a&&m->right==(uint32_t)b)return m;h=(h+1)&(unsigned)(t->hashsize-1);}return NULL;
}
static int bpe(slm_tokenizer *t,const unsigned char *p,int n,int *out,int capacity) {
    if(n<1)return 0;
    if((size_t)n>SIZE_MAX/sizeof(int))return -1;
    int *work=malloc((size_t)n*sizeof(int));if(!work)return -1;
    /* SmolLM2's BPE has no unk token and omits some byte alphabet entries.
     * HF BPE skips those entries. Preserve that exact behavior, including gaps. */
    int count=0;
    for(int i=0;i<n;i++)if(t->byteids[p[i]]>=0)work[count++]=t->byteids[p[i]];
    while(count>1) {
        uint32_t rank=UINT_MAX;int at=-1,merged=-1;
        for(int i=0;i<count-1;i++){Merge *m=lookup(t,work[i],work[i+1]);if(m&&m->rank<rank){rank=m->rank;at=i;merged=(int)m->out;}}
        if(at<0)break;
        work[at]=merged;memmove(work+at+1,work+at+2,(size_t)(count-at-2)*sizeof(int));count--;
    }
    if(count>capacity){free(work);return -1;}memcpy(out,work,(size_t)count*sizeof(int));free(work);return count;
}
/* Exact ordered alternatives in LFM's GPT4-style split regex. */
static uint32_t contraction_lower(uint32_t c){if(c>='A'&&c<='Z')return c+32;if(c==0x017f)return 's';return c;}
static int gpt4_end(const uint32_t *cp,const int *flags,int nc,int i) {
    int end,j;
    if(cp[i]=='\''){
        const char *suffixes[]={"s","t","re","ve","m","ll","d"};
        for(int a=0;a<7;a++){int n=(int)strlen(suffixes[a]);if(i+n<nc){int ok=1;for(int k=0;k<n;k++)if(contraction_lower(cp[i+1+k])!=(unsigned char)suffixes[a][k])ok=0;if(ok)return i+1+n;}}
    }
    j=i;
    if(!(flags[j]&3)&&cp[j]!='\r'&&cp[j]!='\n'&&j+1<nc)j++;
    if(flags[j]&1){end=j+1;while(end<nc&&(flags[end]&1))end++;return end;}
    if(flags[i]&2){end=i+1;while(end<nc&&end<i+3&&(flags[end]&2))end++;return end;}
    j=i;if(cp[j]==' '&&j+1<nc)j++;
    if(!(flags[j]&7)){end=j+1;while(end<nc&&!(flags[end]&7))end++;while(end<nc&&(cp[end]=='\r'||cp[end]=='\n'))end++;return end;}
    end=i;int last_newline=-1;
    while(end<nc&&(flags[end]&4)){if(cp[end]=='\r'||cp[end]=='\n')last_newline=end;end++;}
    if(last_newline>=0)return last_newline+1;
    if(end==nc)return end;
    if(end-i>1)return end-1;
    return end>i?end:i+1;
}
/* Version 1: GPT-2 regex after Digits. Version 2: GPT4 regex on the whole span. */
static int regex_span(slm_tokenizer *t,const unsigned char *s,int bytes,int *out,int capacity) {
    if(bytes==0)return 0;
    if((size_t)bytes>SIZE_MAX/sizeof(uint32_t)-1)return -1;
    int *offset=malloc(((size_t)bytes+1)*sizeof(int)),*flags=malloc(((size_t)bytes+1)*sizeof(int));
    uint32_t *cp=malloc(((size_t)bytes+1)*sizeof(uint32_t));
    if(!offset||!flags||!cp){free(offset);free(flags);free(cp);return -1;}
    int nc=0,b=0,used=0;while(b<bytes){offset[nc]=b;int k=utf8(s+b,bytes-b,cp+nc);flags[nc]=properties(t,cp[nc]);nc++;b+=k;}offset[nc]=bytes;
    for(int i=0;i<nc;) {
        int end=i,matched=0;
        if(t->version==2){end=gpt4_end(cp,flags,nc,i);matched=1;}
        if(!matched&&cp[i]=='\'') {
            const char *suffixes[]={"s","t","re","ve","m","ll","d"};
            for(int a=0;a<7&&!matched;a++){int len=(int)strlen(suffixes[a]);if(i+len<nc){int ok=1;for(int k=0;k<len;k++)if(cp[i+1+k]!=(unsigned char)suffixes[a][k])ok=0;if(ok){end=i+1+len;matched=1;}}}
        }
        if(!matched) {
            int j=i;if(cp[j]==' '&&j+1<nc)j++;
            int cls=(flags[j]&1)?1:(flags[j]&2)?2:!(flags[j]&4)?8:0;
            if(cls){end=j+1;while(end<nc&&((cls==8)?!(flags[end]&7):(flags[end]&cls)))end++;matched=1;}
        }
        if(!matched) {
            end=i+1;while(end<nc&&(flags[end]&4))end++;
            /* The lookahead branch backtracks one whitespace before a non-space. */
            if(end<nc&&end-i>1)end--;
        }
        int n=bpe(t,s+offset[i],offset[end]-offset[i],out+used,capacity-used);
        if(n<0){used=-1;break;}used+=n;i=end;
    }
    free(offset);free(flags);free(cp);return used;
}
static int ordinary(slm_tokenizer *t,const unsigned char *s,int len,int *out,int capacity) {
    if(t->version==2)return regex_span(t,s,len,out,capacity);
    int start=0,pos=0,count=0;
    while(pos<len){uint32_t cp;int n=utf8(s+pos,len-pos,&cp);if(properties(t,cp)&2){int k=regex_span(t,s+start,pos-start,out+count,capacity-count);if(k<0)return -1;count+=k;k=bpe(t,s+pos,n,out+count,capacity-count);if(k<0)return -1;count+=k;start=pos+n;}pos+=n;}
    int k=regex_span(t,s+start,len-start,out+count,capacity-count);return k<0?-1:count+k;
}
int slm_encode(slm_tokenizer *t,const char *text,int specials,int *ids,int capacity) {
    if(!t||!text||!ids||capacity<0)return -1;
    size_t z=strlen(text);if(z>INT_MAX)return -1;int len=(int)z,count=0,start=0;
    if(!specials&&t->version==1)return ordinary(t,(const unsigned char*)text,len,ids,capacity);
    for(int i=0;i<len;) {
        int best=-1,sz=0;
        for(int a=0;a<t->nadded;a++){int j=t->added[a];Token *p=t->tokens+j;if((specials||!(p->special&1))&&p->len>(uint32_t)sz&&p->len<=(uint32_t)(len-i)&&!memcmp(text+i,p->bytes,p->len)){best=j;sz=(int)p->len;}}
        if(best>=0){int n=ordinary(t,(const unsigned char*)text+start,i-start,ids+count,capacity-count);if(n<0||count+n>=capacity)return -1;count+=n;ids[count++]=best;i+=sz;start=i;}else i++;
    }
    int n=ordinary(t,(const unsigned char*)text+start,len-start,ids+count,capacity-count);return n<0?-1:count+n;
}
