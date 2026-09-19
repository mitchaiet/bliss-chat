/* Native SmolLM2/Llama and LFM2 inference for WinXP32 SSE2 and POSIX.
 * Independent C implementation: HF split-half RoPE, learned RMSNorm, GQA, SwiGLU.
 * Q4/Q8 matrices stay mapped, embeddings are dequantized one row at a time.
 * Q4/Q8 inference uses Q8 activations; Q6 uses Q16 activations and SSE2 signed16
 * dot products. --float-activations retains the reference FP32 path.
 */
#define _CRT_SECURE_NO_WARNINGS
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0501
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <errno.h>
#include <ctype.h>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/select.h>
#include <fcntl.h>
#include <unistd.h>
#endif
#if defined(__SSE2__) && !defined(SLM_SCALAR)
#include <emmintrin.h>
#define SIMD 1
#else
#define SIMD 0
#endif
#include "slm_tokenizer.h"
#if defined(SLM_SCALAR) && !defined(SLM_FLOAT_DOT_SCALAR)
#define SLM_FLOAT_DOT_SCALAR
#endif
#include "slm_float_dot.h"
#include "slm_q6_dot.h"
#include "slm_q6x4.h"
#include "slm_sha256.h"
#include "slm_threads.h"
#ifdef _WIN32
#include <sys/stat.h>
#endif
typedef struct {const unsigned char *q; const float *scales; int rows,cols;} Matrix;
typedef struct {const float *an,*fn,*qn,*kn,*conv; Matrix q,k,v,o,in,gate,up,down;int type,cache;} Layer;
typedef struct {
    unsigned char *map;size_t mapsize;int expanded,bits,d,ff,nl,nh,nk,hd,vocab,ctx,group,bos,eos,start,arch,conv_width,na,nc;float theta,eps;
    Matrix emb;Layer *layers;const float *norm;
#ifdef _WIN32
    HANDLE file,mapping;
#else
    int fd;
#endif
} Model;
typedef struct {
    Model *m;float *x,*xb,*tmp,*q,*k,*v,*gate,*up,*att,*logits,*kc,*vc,*rotcos,*rotsin,*conv_state,*conv_proj;
    int8_t *aq;int16_t *aq16,*aq4,*aq4_allocation;int *correction;float *as;int pos;
    /* Prefill batch: several tokens share one pass over the weights. */
    int batch;size_t pair_stride,group_stride;
    float *xB,*xbB,*tmpB,*qB,*kB,*vB,*gateB,*upB,*projB,*asb;
    int16_t *aq4b,*aq4b_allocation;int *corrb;
} State;
static int float_activations=0;
static void die(const char *s){fprintf(stderr,"[slm] %s\n",s);exit(2);}
static void *alloc(size_t count,size_t size){if(size&&count>SIZE_MAX/size)die("allocation overflow");void *p=calloc(count,size);if(!p)die("out of memory");return p;}
static uint32_t u32(const unsigned char *p){uint32_t x;memcpy(&x,p,4);return x;}
static float f32(const unsigned char *p){float x;memcpy(&x,p,4);return x;}
static const unsigned char *take(Model *m,size_t *offset,size_t bytes){if(*offset>m->mapsize||bytes>m->mapsize-*offset)die("truncated model tensor");const unsigned char *p=m->map+*offset;*offset+=bytes;return p;}
static size_t checked_size(uint64_t n){if(n>SIZE_MAX)die("tensor allocation exceeds address space");return(size_t)n;}
static Matrix matrix(Model *m,size_t *off,int rows,int cols){Matrix w;w.rows=rows;w.cols=cols;if(m->expanded&&rows%4)die("Q6X4 requires four-row aligned matrices");uint64_t n=(uint64_t)rows*(uint64_t)cols;w.q=take(m,off,checked_size(n*(unsigned)m->bits/8));w.scales=m->bits==32?NULL:(const float*)take(m,off,checked_size(n/(unsigned)m->group*4));return w;}
static Model *load_model(const char *path,int ctx){
    Model *m=alloc(1,sizeof(*m));
#ifdef _WIN32
    m->file=CreateFileA(path,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);
    if(m->file==INVALID_HANDLE_VALUE)die("cannot open model");
    DWORD high=0,low=GetFileSize(m->file,&high);
    if(high||low==INVALID_FILE_SIZE||low>INT_MAX)die("model must be smaller than2GiB");
    m->mapsize=low;
    m->mapping=CreateFileMappingA(m->file,NULL,PAGE_READONLY,0,0,NULL);if(!m->mapping)die("CreateFileMapping failed");
    m->map=MapViewOfFile(m->mapping,FILE_MAP_READ,0,0,0);if(!m->map)die("MapViewOfFile failed");
#else
    m->fd=open(path,O_RDONLY);if(m->fd<0)die("cannot open model");struct stat st;if(fstat(m->fd,&st)||st.st_size<256||(uint64_t)st.st_size>SIZE_MAX)die("invalid model size");m->mapsize=(size_t)st.st_size;
    m->map=mmap(NULL,m->mapsize,PROT_READ,MAP_PRIVATE,m->fd,0);if(m->map==MAP_FAILED)die("mmap failed");
#endif
    if(m->mapsize<256||memcmp(m->map,"SLMODEL1",8)||(u32(m->map+8)!=1&&u32(m->map+8)!=2&&u32(m->map+8)!=3))die("invalid SLM model header");
    m->expanded=u32(m->map+8)==3;
    m->bits=(int)u32(m->map+12);m->d=(int)u32(m->map+16);m->ff=(int)u32(m->map+20);m->nl=(int)u32(m->map+24);m->nh=(int)u32(m->map+28);m->nk=(int)u32(m->map+32);m->hd=(int)u32(m->map+36);m->vocab=(int)u32(m->map+40);m->ctx=(int)u32(m->map+44);m->group=(int)u32(m->map+48);m->bos=(int)u32(m->map+52);m->eos=(int)u32(m->map+56);m->theta=f32(m->map+64);m->eps=f32(m->map+68);
    if((m->bits!=4&&m->bits!=6&&m->bits!=8&&m->bits!=32)||m->d<64||m->d>4096||m->ff<64||m->ff>16384||m->nl<1||m->nl>128||m->nh<1||m->nh>64||m->nk<1||m->nk>m->nh||m->nh%m->nk||m->hd<2||m->hd>256||m->hd%2||m->nh*m->hd!=m->d||m->vocab<256||m->vocab>200000||m->ctx<32||m->ctx>8192||(m->group!=16&&m->group!=32&&m->group!=64)||m->d%m->group||m->ff%m->group||m->bos<0||m->bos>=m->vocab||m->eos<0||m->eos>=m->vocab||!isfinite(m->theta)||m->theta<=1||!isfinite(m->eps)||m->eps<=0)die("unsupported or corrupt model config");
    m->start=m->bos;
    if(u32(m->map+8)>=2){m->arch=(int)u32(m->map+60);m->conv_width=(int)u32(m->map+72);m->start=(int)u32(m->map+76);if(m->arch!=1||m->conv_width!=3||m->start<0||m->start>=m->vocab)die("unsupported hybrid model header");}
    if(m->expanded&&(m->bits!=8||m->group!=64||!m->arch))die("invalid Q6X4 model config");
    if(m->bits==6&&(!m->arch||m->group!=64))die("Q6 requires an LFM model with group64");
    if(ctx>0){if(ctx>m->ctx||ctx<32)die("requested context outside model range");m->ctx=ctx;}
    size_t off=256;m->emb=matrix(m,&off,m->vocab,m->d);m->layers=alloc((size_t)m->nl,sizeof(Layer));
    for(int l=0;l<m->nl;l++){
        Layer *w=m->layers+l;w->type=m->arch?m->map[80+l]:0;if(w->type>1)die("invalid hybrid layer type");
        w->an=(const float*)take(m,&off,(size_t)m->d*4);w->fn=(const float*)take(m,&off,(size_t)m->d*4);
        if(w->type){w->cache=m->nc++;w->conv=(const float*)take(m,&off,(size_t)m->d*m->conv_width*4);w->in=matrix(m,&off,3*m->d,m->d);w->o=matrix(m,&off,m->d,m->d);}
        else{w->cache=m->na++;if(m->arch){w->qn=(const float*)take(m,&off,(size_t)m->hd*4);w->kn=(const float*)take(m,&off,(size_t)m->hd*4);}w->q=matrix(m,&off,m->d,m->d);w->k=matrix(m,&off,m->nk*m->hd,m->d);w->v=matrix(m,&off,m->nk*m->hd,m->d);w->o=matrix(m,&off,m->d,m->d);}
        w->gate=matrix(m,&off,m->ff,m->d);w->up=matrix(m,&off,m->ff,m->d);w->down=matrix(m,&off,m->d,m->ff);
    }
    m->norm=(const float*)take(m,&off,(size_t)m->d*4);if(off!=m->mapsize)die("unexpected trailing model bytes");return m;
}
static void free_model(Model *m){free(m->layers);
#ifdef _WIN32
    UnmapViewOfFile(m->map);CloseHandle(m->mapping);CloseHandle(m->file);
#else
    munmap(m->map,m->mapsize);close(m->fd);
#endif
    free(m);
}
static State *new_state(Model *m){State *s=alloc(1,sizeof(*s));s->m=m;int d=m->d,kd=m->nk*m->hd;
    size_t kv_elements=checked_size((uint64_t)m->na*(unsigned)m->ctx*(unsigned)kd);int widest=d>m->ff?d:m->ff;
    s->x=alloc((size_t)d,4);s->xb=alloc((size_t)d,4);s->tmp=alloc((size_t)d,4);s->q=alloc((size_t)d,4);s->k=alloc((size_t)kd,4);s->v=alloc((size_t)kd,4);s->gate=alloc((size_t)m->ff,4);s->up=alloc((size_t)m->ff,4);s->att=alloc((size_t)m->ctx*(size_t)m->nh,4);s->logits=alloc((size_t)m->vocab,4);s->kc=alloc(kv_elements?kv_elements:1,4);s->vc=alloc(kv_elements?kv_elements:1,4);s->aq=alloc((size_t)widest,1);s->aq16=alloc((size_t)widest,2);s->as=alloc((size_t)widest/m->group,4);
    if(m->expanded){s->aq4_allocation=alloc((size_t)widest*4+7,2);s->aq4=(int16_t*)(((uintptr_t)s->aq4_allocation+15)&~(uintptr_t)15);s->correction=alloc((size_t)widest/64,4);}
    /* Batch buffers, about 840 KiB at the default of eight tokens. Only the
     * expanded Q6X4 path has a batched kernel, so nothing else allocates them. */
    s->batch=1;
    if(m->expanded&&!float_activations){
        const char *env=getenv("SLM_BATCH");int want=env?atoi(env):8;
        if(want<1)want=1;if(want>SLM_Q6X4_MAX_BATCH)want=SLM_Q6X4_MAX_BATCH;
        s->batch=want;
    }
    if(s->batch>1){
        size_t n=(size_t)s->batch;
        s->pair_stride=(size_t)widest*4;s->group_stride=(size_t)widest/64;
        s->xB=alloc(n*(size_t)d,4);s->xbB=alloc(n*(size_t)d,4);s->tmpB=alloc(n*(size_t)d,4);
        s->qB=alloc(n*(size_t)d,4);s->kB=alloc(n*(size_t)kd,4);s->vB=alloc(n*(size_t)kd,4);
        s->gateB=alloc(n*(size_t)m->ff,4);s->upB=alloc(n*(size_t)m->ff,4);
        s->projB=alloc(m->nc?n*3*(size_t)d:1,4);
        s->aq4b_allocation=alloc(n*s->pair_stride+8,2);
        s->aq4b=(int16_t*)(((uintptr_t)s->aq4b_allocation+15)&~(uintptr_t)15);
        s->asb=alloc(n*s->group_stride,4);s->corrb=alloc(n*s->group_stride,4);
    }
    if(m->nc){s->conv_state=alloc(checked_size((uint64_t)m->nc*(unsigned)d*(unsigned)m->conv_width),4);s->conv_proj=alloc((size_t)3*d,4);}
    s->rotcos=alloc((size_t)m->ctx*m->hd/2,4);s->rotsin=alloc((size_t)m->ctx*m->hd/2,4);
    for(int p=0;p<m->ctx;p++)for(int j=0;j<m->hd/2;j++){float angle=(float)p/powf(m->theta,(float)(2*j)/(float)m->hd);s->rotcos[p*(m->hd/2)+j]=cosf(angle);s->rotsin[p*(m->hd/2)+j]=sinf(angle);}
    return s;
}
static void reset_state(State *s){s->pos=0;if(s->conv_state)memset(s->conv_state,0,checked_size((uint64_t)s->m->nc*(unsigned)s->m->d*(unsigned)s->m->conv_width*4));}
static void free_state(State *s){free(s->x);free(s->xb);free(s->tmp);free(s->q);free(s->k);free(s->v);free(s->gate);free(s->up);free(s->att);free(s->logits);free(s->kc);free(s->vc);free(s->aq);free(s->aq16);free(s->aq4_allocation);free(s->correction);free(s->as);free(s->rotcos);free(s->rotsin);free(s->conv_state);free(s->conv_proj);
    free(s->xB);free(s->xbB);free(s->tmpB);free(s->qB);free(s->kB);free(s->vB);
    free(s->gateB);free(s->upB);free(s->projB);free(s->aq4b_allocation);free(s->asb);free(s->corrb);free(s);}
static float dot(const float *a,const float *b,int n){float sum=0;
#if SIMD
    __m128 z=_mm_setzero_ps();int i=0;for(;i+4<=n;i+=4)z=_mm_add_ps(z,_mm_mul_ps(_mm_loadu_ps(a+i),_mm_loadu_ps(b+i)));float v[4];_mm_storeu_ps(v,z);sum=v[0]+v[1]+v[2]+v[3];for(;i<n;i++)sum+=a[i]*b[i];
#else
    for(int i=0;i<n;i++)sum+=a[i]*b[i];
#endif
    return sum;
}
static void axpy(float *out,const float *x,float a,int n){
#if SIMD
    __m128 av=_mm_set1_ps(a);int i=0;for(;i+4<=n;i+=4)_mm_storeu_ps(out+i,_mm_add_ps(_mm_loadu_ps(out+i),_mm_mul_ps(av,_mm_loadu_ps(x+i))));for(;i<n;i++)out[i]+=a*x[i];
#else
    for(int i=0;i<n;i++)out[i]+=a*x[i];
#endif
}
static void rms(float *out,const float *x,const float *w,int n,float eps){float r=1.0f/sqrtf(dot(x,x,n)/(float)n+eps);for(int j=0;j<n;j++)out[j]=x[j]*r*w[j];}
static void quant_activation(State *s,const float *x,int n){int width=s->m->group;for(int g=0;g<n/width;g++){float a=0;for(int j=0;j<width;j++){float z=fabsf(x[g*width+j]);if(z>a)a=z;}float scale=a>0?a/127.0f:1.0f;s->as[g]=scale;for(int j=0;j<width;j++){long q=lrintf(x[g*width+j]/scale);if(q>127)q=127;if(q< -127)q=-127;s->aq[g*width+j]=(int8_t)q;}}}
static int dot_quant(const unsigned char *w,const int8_t *x,int bits,int n){
#if SIMD
    __m128i accum=_mm_setzero_si128(),zero=_mm_setzero_si128();
    for(int j=0;j<n;j+=16){
        __m128i wb;
        if(bits==4){__m128i p=_mm_loadl_epi64((const __m128i*)(w+j/2)),mask=_mm_set1_epi8(15);wb=_mm_unpacklo_epi8(_mm_and_si128(p,mask),_mm_and_si128(_mm_srli_epi16(p,4),mask));wb=_mm_sub_epi8(wb,_mm_set1_epi8(8));}
        else wb=_mm_loadu_si128((const __m128i*)(w+j));
        __m128i xb=_mm_loadu_si128((const __m128i*)(x+j)),ws=_mm_cmpgt_epi8(zero,wb),xs=_mm_cmpgt_epi8(zero,xb);
        accum=_mm_add_epi32(accum,_mm_madd_epi16(_mm_unpacklo_epi8(wb,ws),_mm_unpacklo_epi8(xb,xs)));
        accum=_mm_add_epi32(accum,_mm_madd_epi16(_mm_unpackhi_epi8(wb,ws),_mm_unpackhi_epi8(xb,xs)));
    }
    int v[4];_mm_storeu_si128((__m128i*)v,accum);return v[0]+v[1]+v[2]+v[3];
#else
    int z=0;for(int j=0;j<n;j++){int q=bits==4?((w[j/2]>>(4*(j%2)))&15)-8:(int)((const int8_t*)w)[j];z+=q*x[j];}return z;
#endif
}
/* One contiguous range of output rows. Activations were already quantized by
 * linear(), so every thread reads the same prepared vectors and writes only
 * its own rows; the arithmetic per row is unchanged from the serial version. */
typedef struct {State *s;float *out;const Matrix *w;const float *x;} LinearJob;
static void linear_rows(void *job_,int r0,int r1){
    LinearJob *job=(LinearJob*)job_;State *s=job->s;Model *m=s->m;float *out=job->out;const Matrix *w=job->w;const float *x=job->x;
    if(m->bits==32){const float *p=(const float*)w->q;for(int r=r0;r<r1;r++)out[r]=dot(p+(size_t)r*w->cols,x,w->cols);return;}
    if(m->expanded){
        int ng=w->cols/64;
        if(float_activations){
            for(int r=r0;r<r1;r++){
                float total=0;
                for(int g=0;g<ng;g++){
                    int8_t q[64];
                    for(int j=0;j<64;j++)q[j]=(int8_t)slm_q6x4_value(w->q,ng,r,g*64+j);
                    total+=slm_dot_float((const unsigned char*)q,x+g*64,8,64)*w->scales[(size_t)(r/4)*ng*4+g*4+r%4];
                }out[r]=total;
            }
        }else{
            /* Ranges are multiples of four rows, matching the Q6X4 block layout. */
            slm_q6x4_linear(out+r0,w->q+(size_t)(r0/4)*ng*256,w->scales+(size_t)r0*ng,r1-r0,ng,s->aq4,s->as,s->correction);
        }return;
    }
    if(m->bits==6&&!float_activations){
        int ng=w->cols/64;
        for(int r=r0;r<r1;r++){
            const unsigned char *row=w->q+(size_t)r*ng*48;
            const float *scales=w->scales+(size_t)r*ng;
            float total=0;
            for(int g=0;g<ng;g++)total+=(float)slm_dot_q6_q16(row+(size_t)g*48,s->aq16+g*64)*scales[g]*s->as[g];
            out[r]=total;
        }
        return;
    }
    if(float_activations||m->bits==6){
        int ng=w->cols/m->group,gb=m->group*m->bits/8;
        for(int r=r0;r<r1;r++){
            float total=0;const unsigned char *row=w->q+(size_t)r*(size_t)ng*(size_t)gb;
            for(int g=0;g<ng;g++)total+=slm_dot_float(row+(size_t)g*gb,x+g*m->group,m->bits,m->group)*w->scales[(size_t)r*ng+g];
            out[r]=total;
        }return;
    }
    int ng=w->cols/m->group,gb=m->group*m->bits/8;
    for(int r=r0;r<r1;r++){float z=0;const unsigned char *p=w->q+(size_t)r*(size_t)ng*(size_t)gb;const float *scale=w->scales+(size_t)r*ng;for(int g=0;g<ng;g++)z+=(float)dot_quant(p+(size_t)g*gb,s->aq+g*m->group,m->bits,m->group)*scale[g]*s->as[g];out[r]=z;}
}
static void linear(State *s,float *out,const Matrix *w,const float *x,int already_quant){Model *m=s->m;int align=1;
    if(m->bits!=32&&!float_activations){
        if(m->expanded){if(!already_quant)slm_q6x4_prepare(x,s->aq16,s->aq4,s->as,s->correction,w->cols/64);align=8;}
        else if(m->bits==6){if(!already_quant)for(int g=0;g<w->cols/64;g++)slm_quantize_q16_64(x+g*64,s->aq16+g*64,s->as+g);}
        else if(!already_quant)quant_activation(s,x,w->cols);
    }
    LinearJob job={s,out,w,x};slm_parallel_rows(w->rows,align,linear_rows,&job);
}
/* Batched linear for the expanded Q6X4 path. Activations for every token in the
 * batch are quantized first, then one sweep of the weights serves them all. The
 * single-token path above is left untouched. */
typedef struct {State *s;float *out;size_t out_stride;const Matrix *w;int batch;} BatchLinearJob;
static void linear_batch_rows(void *job_,int r0,int r1){
    BatchLinearJob *job=(BatchLinearJob*)job_;State *s=job->s;const Matrix *w=job->w;int ng=w->cols/64;
    slm_q6x4_linear_batch(job->out+r0,job->out_stride,
        w->q+(size_t)(r0/4)*ng*256,w->scales+(size_t)r0*ng,r1-r0,ng,
        s->aq4b,s->pair_stride,s->asb,s->group_stride,s->corrb,s->group_stride,job->batch);
}
static void linear_batch(State *s,float *out,size_t out_stride,const Matrix *w,
                         const float *x,size_t x_stride,int batch,int already_quant){
    if(!already_quant)
        for(int b=0;b<batch;b++)
            slm_q6x4_prepare(x+(size_t)b*x_stride,s->aq16,
                s->aq4b+(size_t)b*s->pair_stride,s->asb+(size_t)b*s->group_stride,
                s->corrb+(size_t)b*s->group_stride,w->cols/64);
    {BatchLinearJob job={s,out,out_stride,w,batch};
     slm_parallel_rows(w->rows,8,linear_batch_rows,&job);}
}
/* These take explicit buffers so one token and a whole batch run the same
 * arithmetic through the same code; only the pointers and position differ. */
static void embedding(State *s,int token,float *out){Model *m=s->m;size_t base=(size_t)token*m->d;
    if(m->bits==32){memcpy(out,(const float*)m->emb.q+base,(size_t)m->d*4);return;}
    for(int j=0;j<m->d;j++){size_t i=base+(size_t)j;int q=m->expanded?slm_q6x4_value(m->emb.q,m->d/64,token,j):m->bits==6?slm_q6_value(m->emb.q+(i/64)*48,(int)(i%64)):m->bits==4?((m->emb.q[i/2]>>(4*(i%2)))&15)-8:((const int8_t*)m->emb.q)[i];out[j]=(float)q*m->emb.scales[m->expanded?((size_t)(token/4)*(m->d/64)+j/64)*4+token%4:i/(size_t)m->group];}
}
static void rope(State *s,float *x,int heads,int pos){int hd=s->m->hd,half=hd/2;const float *co=s->rotcos+(size_t)pos*half,*si=s->rotsin+(size_t)pos*half;for(int h=0;h<heads;h++)for(int j=0;j<half;j++){float a=x[h*hd+j],b=x[h*hd+j+half];x[h*hd+j]=a*co[j]-b*si[j];x[h*hd+j+half]=b*co[j]+a*si[j];}}
/* Optional section timing for the XP benchmark (-DSLM_PROFILE). */
#ifdef SLM_PROFILE
static double slm_prof[8];
static double slm_prof_now(void){
#ifdef _WIN32
    LARGE_INTEGER c;QueryPerformanceCounter(&c);return (double)c.QuadPart;
#else
    struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec*1e9+(double)t.tv_nsec;
#endif
}
#define PROF_START double prof_t0=slm_prof_now();
#define PROF_STOP(i) slm_prof[i]+=slm_prof_now()-prof_t0;
#define PROF_LAP(i) slm_prof[i]+=slm_prof_now()-prof_t0;prof_t0=slm_prof_now();
#else
#define PROF_START
#define PROF_STOP(i)
#define PROF_LAP(i)
#endif
/* Parallel pieces of the forward pass. Each job writes disjoint elements and
 * repeats the serial arithmetic exactly, so outputs stay bit-identical. */
typedef struct {State *s;Layer *w;int pos;const float *q;float *out;} AttnJob;
static void attention_heads(void *job_,int h0,int h1){
    AttnJob *job=(AttnJob*)job_;State *s=job->s;Model *m=s->m;Layer *w=job->w;int pos=job->pos,kd=m->nk*m->hd;
    float *kc=s->kc+(size_t)w->cache*m->ctx*kd,*vc=s->vc+(size_t)w->cache*m->ctx*kd;
    for(int h=h0;h<h1;h++){
        int kh=h/(m->nh/m->nk);float mx=-INFINITY;float *att=s->att+(size_t)h*m->ctx;
        for(int t=0;t<=pos;t++){float z=dot(job->q+h*m->hd,kc+(size_t)t*kd+kh*m->hd,m->hd)/sqrtf((float)m->hd);att[t]=z;if(z>mx)mx=z;}
        float sum=0;for(int t=0;t<=pos;t++){att[t]=expf(att[t]-mx);sum+=att[t];}
        for(int t=0;t<=pos;t++)axpy(job->out+h*m->hd,vc+(size_t)t*kd+kh*m->hd,att[t]/sum,m->hd);
    }
}
typedef struct {State *s;Layer *w;const float *proj;float *out;} ConvJob;
static void conv_channels(void *job_,int j0,int j1){
    ConvJob *job=(ConvJob*)job_;State *s=job->s;Model *m=s->m;Layer *w=job->w;int d=m->d;
    float *cache=s->conv_state+(size_t)w->cache*d*m->conv_width;
    for(int j=j0;j<j1;j++){
        float *history=cache+(size_t)j*m->conv_width;
        for(int k=0;k<m->conv_width-1;k++)history[k]=history[k+1];
        history[m->conv_width-1]=job->proj[j]*job->proj[2*d+j];
        /* PyTorch Conv1d is cross-correlation: oldest sample uses kernel[0]. */
        float z=0;for(int k=0;k<m->conv_width;k++)z+=history[k]*w->conv[(size_t)j*m->conv_width+k];
        job->out[j]=job->proj[d+j]*z;
    }
}
typedef struct {float *gate;const float *up;} SwigluJob;
static void swiglu_range(void *job_,int j0,int j1){
    SwigluJob *job=(SwigluJob*)job_;for(int j=j0;j<j1;j++)job->gate[j]=(job->gate[j]/(1.0f+expf(-job->gate[j])))*job->up[j];
}
static float *forward(State *s,int token,int logits){
    Model *m=s->m;if(token<0||token>=m->vocab||s->pos>=m->ctx)die("token or context out of bounds");
    int d=m->d,kd=m->nk*m->hd,pos=s->pos;
    PROF_START
    embedding(s,token,s->x);
    for(int l=0;l<m->nl;l++){
        Layer *w=m->layers+l;rms(s->xb,s->x,w->an,d,m->eps);
        PROF_LAP(4)
        if(w->type){
            linear(s,s->conv_proj,&w->in,s->xb,0);
            PROF_LAP(0)
            ConvJob cj={s,w,s->conv_proj,s->xb};slm_parallel_rows(d,64,conv_channels,&cj);
            PROF_LAP(3)
        }else{
            linear(s,s->q,&w->q,s->xb,0);linear(s,s->k,&w->k,s->xb,1);linear(s,s->v,&w->v,s->xb,1);
            PROF_LAP(0)
            if(w->qn){for(int h=0;h<m->nh;h++)rms(s->q+h*m->hd,s->q+h*m->hd,w->qn,m->hd,m->eps);for(int h=0;h<m->nk;h++)rms(s->k+h*m->hd,s->k+h*m->hd,w->kn,m->hd,m->eps);}
            rope(s,s->q,m->nh,pos);rope(s,s->k,m->nk,pos);
            float *kc=s->kc+(size_t)w->cache*m->ctx*kd,*vc=s->vc+(size_t)w->cache*m->ctx*kd;
            memcpy(kc+(size_t)pos*kd,s->k,(size_t)kd*4);memcpy(vc+(size_t)pos*kd,s->v,(size_t)kd*4);memset(s->xb,0,(size_t)d*4);
            PROF_LAP(4)
            AttnJob aj={s,w,pos,s->q,s->xb};slm_parallel_rows(m->nh,1,attention_heads,&aj);
            PROF_LAP(1)
        }
        linear(s,s->tmp,&w->o,s->xb,0);
        PROF_LAP(0)
        axpy(s->x,s->tmp,1,d);
        rms(s->xb,s->x,w->fn,d,m->eps);
        PROF_LAP(4)
        linear(s,s->gate,&w->gate,s->xb,0);linear(s,s->up,&w->up,s->xb,1);
        PROF_LAP(0)
        {SwigluJob sj={s->gate,s->up};slm_parallel_rows(m->ff,64,swiglu_range,&sj);}
        PROF_LAP(2)
        linear(s,s->tmp,&w->down,s->gate,0);
        PROF_LAP(0)
        axpy(s->x,s->tmp,1,d);
        PROF_LAP(4)
    }
    if(logits){rms(s->xb,s->x,m->norm,d,m->eps);PROF_LAP(4) linear(s,s->logits,&m->emb,s->xb,0);PROF_LAP(0)}
    s->pos++;return s->logits;
}
/* Prefill several tokens in one sweep of the weights. Reading a prompt used to
 * cost a full pass over the model per token; a batch pays that once and reuses
 * each weight block from cache for the rest. Every token runs the same
 * arithmetic in the same order as forward() above, so the state this leaves
 * behind is identical to feeding the tokens one at a time. */
static float *forward_batch(State *s,const int *tokens,int n,int logits){
    Model *m=s->m;int d=m->d,kd=m->nk*m->hd,base=s->pos,b;
    if(!m->expanded||float_activations||n<1||n>s->batch)die("prefill batch unavailable");
    if(base+n>m->ctx)die("token or context out of bounds");
    for(b=0;b<n;b++){
        if(tokens[b]<0||tokens[b]>=m->vocab)die("token or context out of bounds");
        embedding(s,tokens[b],s->xB+(size_t)b*d);
    }
    for(int l=0;l<m->nl;l++){
        Layer *w=m->layers+l;
        for(b=0;b<n;b++)rms(s->xbB+(size_t)b*d,s->xB+(size_t)b*d,w->an,d,m->eps);
        if(w->type){
            linear_batch(s,s->projB,(size_t)3*d,&w->in,s->xbB,(size_t)d,n,0);
            /* The convolution carries state from one position to the next, so
             * its channels run in order even though the projection was batched. */
            for(b=0;b<n;b++){
                ConvJob cj={s,w,s->projB+(size_t)b*3*d,s->xbB+(size_t)b*d};
                slm_parallel_rows(d,64,conv_channels,&cj);
            }
        }else{
            linear_batch(s,s->qB,(size_t)d,&w->q,s->xbB,(size_t)d,n,0);
            linear_batch(s,s->kB,(size_t)kd,&w->k,s->xbB,(size_t)d,n,1);
            linear_batch(s,s->vB,(size_t)kd,&w->v,s->xbB,(size_t)d,n,1);
            float *kc=s->kc+(size_t)w->cache*m->ctx*kd,*vc=s->vc+(size_t)w->cache*m->ctx*kd;
            for(b=0;b<n;b++){
                float *q=s->qB+(size_t)b*d,*k=s->kB+(size_t)b*kd;
                if(w->qn){for(int h=0;h<m->nh;h++)rms(q+h*m->hd,q+h*m->hd,w->qn,m->hd,m->eps);for(int h=0;h<m->nk;h++)rms(k+h*m->hd,k+h*m->hd,w->kn,m->hd,m->eps);}
                rope(s,q,m->nh,base+b);rope(s,k,m->nk,base+b);
                memcpy(kc+(size_t)(base+b)*kd,k,(size_t)kd*4);
                memcpy(vc+(size_t)(base+b)*kd,s->vB+(size_t)b*kd,(size_t)kd*4);
            }
            /* Every key and value is in place before any attention runs, so
             * each token still attends to exactly its own past and itself. */
            for(b=0;b<n;b++){
                float *out=s->xbB+(size_t)b*d;memset(out,0,(size_t)d*4);
                AttnJob aj={s,w,base+b,s->qB+(size_t)b*d,out};
                slm_parallel_rows(m->nh,1,attention_heads,&aj);
            }
        }
        linear_batch(s,s->tmpB,(size_t)d,&w->o,s->xbB,(size_t)d,n,0);
        for(b=0;b<n;b++)axpy(s->xB+(size_t)b*d,s->tmpB+(size_t)b*d,1,d);
        for(b=0;b<n;b++)rms(s->xbB+(size_t)b*d,s->xB+(size_t)b*d,w->fn,d,m->eps);
        linear_batch(s,s->gateB,(size_t)m->ff,&w->gate,s->xbB,(size_t)d,n,0);
        linear_batch(s,s->upB,(size_t)m->ff,&w->up,s->xbB,(size_t)d,n,1);
        for(b=0;b<n;b++){SwigluJob sj={s->gateB+(size_t)b*m->ff,s->upB+(size_t)b*m->ff};slm_parallel_rows(m->ff,64,swiglu_range,&sj);}
        linear_batch(s,s->tmpB,(size_t)d,&w->down,s->gateB,(size_t)m->ff,n,0);
        for(b=0;b<n;b++)axpy(s->xB+(size_t)b*d,s->tmpB+(size_t)b*d,1,d);
    }
    memcpy(s->x,s->xB+(size_t)(n-1)*d,(size_t)d*4);
    s->pos=base+n;
    if(logits){rms(s->xb,s->x,m->norm,d,m->eps);linear(s,s->logits,&m->emb,s->xb,0);}
    return s->logits;
}
/* Feed a run of known tokens, batching where the model supports it. */
static void prefill(State *s,const int *tokens,int n,int last_logits){
    for(int j=0;j<n;){
        int room=n-j,chunk=s->batch<room?s->batch:room,want=last_logits&&j+chunk==n;
        if(chunk>1)forward_batch(s,tokens+j,chunk,want);
        else forward(s,tokens[j],want);
        j+=chunk;
    }
}
typedef struct {float p;int id;} Candidate;
static int probcmp(const void *a,const void *b){float x=((const Candidate*)a)->p,y=((const Candidate*)b)->p;return x<y?1:x>y?-1:0;}
static uint32_t rngstate=42;
static float uniform(void){rngstate^=rngstate<<13;rngstate^=rngstate>>17;rngstate^=rngstate<<5;return(float)(rngstate>>8)/16777216.0f;}
static int sample(State *s,float temp,float topp,Candidate *c){int n=s->m->vocab,best=0;float max=s->logits[0];for(int i=1;i<n;i++)if(s->logits[i]>max){max=s->logits[i];best=i;}if(temp<=0)return best;float sum=0;for(int i=0;i<n;i++){c[i].id=i;c[i].p=expf((s->logits[i]-max)/temp);sum+=c[i].p;}qsort(c,(size_t)n,sizeof(*c),probcmp);float covered=0;int k=0;do{covered+=c[k++].p;}while(k<n&&covered<topp*sum);float r=uniform()*covered;for(int i=0;i<k;i++){r-=c[i].p;if(r<=0)return c[i].id;}return c[k-1].id;}
static char pending[32768];
static int stop_requested(void){if(pending[0])return 0;int ready=0;
#ifdef _WIN32
    DWORD bytes=0;HANDLE h=GetStdHandle(STD_INPUT_HANDLE);if(GetFileType(h)==FILE_TYPE_PIPE&&PeekNamedPipe(h,NULL,0,NULL,&bytes,NULL)&&bytes)ready=1;
#else
    fd_set rd;FD_ZERO(&rd);FD_SET(0,&rd);struct timeval tv={0,0};ready=select(1,&rd,NULL,NULL,&tv)>0;
#endif
    if(!ready)return 0;
    if(!fgets(pending,sizeof(pending),stdin))return 0;
    if(!strncmp(pending,"\001STOP",5)){pending[0]=0;return 1;}return 0;
}
static void unescape(char *s){char *o=s;while(*s){if(s[0]=='\\'&&s[1]=='n'){*o++='\n';s+=2;}else if(s[0]=='\\'&&s[1]=='t'){*o++='\t';s+=2;}else *o++=*s++;}*o=0;}
/* Existing Bliss MEMORY.TXT is twelve short UTF-8 notes, one per line. */
#define MEM_MAX 12
#define MEM_LINE 160
typedef struct {char path[1024],notes[MEM_MAX][MEM_LINE];int count;} Memories;
static Memories memories;
static void clean_note(char *out,const char *in){int n=0;while(*in&&n<MEM_LINE-1){unsigned char c=(unsigned char)*in++;if(c<32){if(c=='\t'||c=='\r'||c=='\n')c=' ';else continue;}out[n++]=(char)c;}out[n]=0;while(n>0&&(unsigned char)out[n-1]>=0x80){int start=n-1;while(start>0&&((unsigned char)out[start]&0xc0)==0x80)start--;unsigned char first=(unsigned char)out[start];int width=first<0xe0?2:first<0xf0?3:4;if(n-start<width){n=start;out[n]=0;}break;}while(n>0&&out[n-1]==' ')out[--n]=0;int i=0;while(out[i]==' ')i++;if(i)memmove(out,out+i,strlen(out+i)+1);}
static int memory_load(void){Memories next=memories;next.count=0;if(!next.path[0])return 1;FILE *f=fopen(next.path,"rb");if(!f){if(errno==ENOENT){memories.count=0;return 1;}return 0;}char line[1024];while(next.count<MEM_MAX&&fgets(line,sizeof(line),f)){clean_note(next.notes[next.count],line);if(next.notes[next.count][0])next.count++;}int okay=!ferror(f);fclose(f);if(okay)memories=next;return okay;}
static int memory_save(void){if(!memories.path[0])return 0;char path[1100];
#ifdef _WIN32
    unsigned long pid=(unsigned long)GetCurrentProcessId();
#else
    unsigned long pid=(unsigned long)getpid();
#endif
    snprintf(path,sizeof(path),"%s.tmp.%lu",memories.path,pid);FILE *f=fopen(path,"wb");if(!f)return 0;int okay=1;for(int i=0;i<memories.count;i++)if(fprintf(f,"%s\n",memories.notes[i])<0)okay=0;if(fclose(f))okay=0;
    if(okay){
#ifdef _WIN32
        okay=MoveFileExA(path,memories.path,MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)!=0;
#else
        okay=rename(path,memories.path)==0;
#endif
    }if(!okay)remove(path);return okay;
}
static int contains_word(const char *text,const char *word){size_t n=strlen(word);for(size_t i=0;text[i];i++){if(i&&isalnum((unsigned char)text[i-1]))continue;size_t j=0;while(j<n&&text[i+j]&&tolower((unsigned char)text[i+j])==word[j])j++;if(j==n&&!isalnum((unsigned char)text[i+j]))return 1;}return 0;}
static int note_score(const char *query,const char *note){int score=0;const char *stop="a an the is are was were be to of in on at for with and or my i me you your what which how do does did this that have has it from about can tell";while(*query){while(*query&&!isalnum((unsigned char)*query))query++;char word[64];int n=0;while(*query&&isalnum((unsigned char)*query)){if(n<63)word[n++]=(char)tolower((unsigned char)*query);query++;}word[n]=0;if(n>1&&!contains_word(stop,word)&&contains_word(note,word))score++;}return score;}
static char *memory_context(const char *query){size_t cap=strlen(query)+MEM_MAX*MEM_LINE+64;char *text=alloc(cap,1);int picked[MEM_MAX]={0},used=0;for(int take_index=0;take_index<3;take_index++){int best=-1,score=0;for(int i=0;i<memories.count;i++)if(!picked[i]){int v=note_score(query,memories.notes[i]);if(v>score){score=v;best=i;}}if(best<0)break;picked[best]=1;if(!used)strcat(text,"Saved notes:\n");strcat(text,"- ");strcat(text,memories.notes[best]);strcat(text,"\n");used++;}if(used)strcat(text,"\n");strcat(text,query);return text;}
static int encode_text(slm_tokenizer *t,const char *text,int *out,int cap){int n=slm_encode(t,text,0,out,cap);if(n<0)die("input tokenization exceeds capacity");return n;}
typedef struct {
    State *s;slm_tokenizer *tok;int *history,*scratch;
    int used,prefix,ends[128],turns;char system[4096];
    /* Only the system-prefix rows are saved, never the entire context cache. */
    int *prefix_ids,prefix_tokens;float *prefix_state;
    const char *prefix_path;unsigned char prefix_identity[64];
} Chat;
static int role_message(Chat *c,const char *role,const char *text,int closed,int *out,int cap){int n=0;if(cap<1)return -1;out[n++]=c->s->m->start;size_t bytes=strlen(role)+strlen(text)+2;char *body=alloc(bytes,1);snprintf(body,bytes,"%s\n%s",role,text);int k=slm_encode(c->tok,body,0,out+n,cap-n);free(body);if(k<0)return -1;n+=k;if(closed){if(n>=cap)return -1;out[n++]=c->s->m->eos;k=slm_encode(c->tok,"\n",0,out+n,cap-n);if(k<0)return -1;n+=k;}return n;}
static void append(Chat *c,const int *tokens,int n,int last_logits){
    if(c->used+n>c->s->m->ctx)die("history overflow");
    for(int j=0;j<n;j++)c->history[c->used++]=tokens[j];
    prefill(c->s,tokens,n,last_logits);
}
static int system_message(Chat *c,const char *text,int *out,int cap){
    int prefix=c->s->m->arch?1:0;if(cap<=prefix)return -1;
    if(prefix){out[0]=c->s->m->bos;if(!*text)return prefix;}
    int n=role_message(c,"system",text,1,out+prefix,cap-prefix);return n<0?-1:prefix+n;
}
static void free_prefix(Chat *c){
    free(c->prefix_ids);free(c->prefix_state);
    c->prefix_ids=NULL;c->prefix_state=NULL;c->prefix_tokens=0;
}
static int prefix_matches(const Chat *c,const int *ids,int n){
    return c->prefix_ids&&c->prefix_tokens==n&&!memcmp(c->prefix_ids,ids,(size_t)n*sizeof(int));
}
/* Copy exact cached floats, including every convolution delay element. Scratch
 * vectors and logits need no snapshot: the next forward call overwrites them.
 * Rows beyond the prefix remain untouched and are overwritten before use. */
static void transfer_prefix(Chat *c,int restore){
    Model *m=c->s->m;size_t kd=(size_t)m->nk*m->hd;
    size_t rows=(size_t)c->prefix_tokens*kd,layer_stride=(size_t)m->ctx*kd;
    float *saved=c->prefix_state;
    for(int layer=0;layer<m->na;layer++){
        float *key=c->s->kc+(size_t)layer*layer_stride;
        float *value=c->s->vc+(size_t)layer*layer_stride;
        if(restore){memcpy(key,saved,rows*sizeof(float));memcpy(value,saved+rows,rows*sizeof(float));}
        else{memcpy(saved,key,rows*sizeof(float));memcpy(saved+rows,value,rows*sizeof(float));}
        saved+=2*rows;
    }
    size_t conv=(size_t)m->nc*m->d*m->conv_width;
    if(conv){
        if(restore)memcpy(c->s->conv_state,saved,conv*sizeof(float));
        else memcpy(saved,c->s->conv_state,conv*sizeof(float));
    }
    if(restore)c->s->pos=c->prefix_tokens;
}
static void save_prefix(Chat *c){
    Model *m=c->s->m;int n=c->prefix;
    size_t elements=checked_size(2ull*m->na*(unsigned)n*m->nk*m->hd+(uint64_t)m->nc*m->d*m->conv_width);
    c->prefix_ids=alloc((size_t)n,sizeof(int));
    c->prefix_state=alloc(elements?elements:1,sizeof(float));
    c->prefix_tokens=n;
    memcpy(c->prefix_ids,c->history,(size_t)n*sizeof(int));transfer_prefix(c,0);
}
#include "slm_prefix_disk.h"
static int chat_reset(Chat *c){
    int n=system_message(c,c->system,c->scratch,c->s->m->ctx);
    if(n<0||n>c->s->m->ctx/2)return 0;
    c->used=0;c->turns=0;
    if(prefix_matches(c,c->scratch,n)){
        transfer_prefix(c,1);memcpy(c->history,c->prefix_ids,(size_t)n*sizeof(int));c->used=n;
    }else{
        free_prefix(c);
        if(!prefix_disk_load(c,c->scratch,n)){
            reset_state(c->s);append(c,c->scratch,n,0);
            c->prefix=n;save_prefix(c);prefix_disk_save(c);
        }
    }
    c->prefix=n;return 1;
}
static int make_room(Chat *c,int needed){
    int ctx=c->s->m->ctx;if(needed>ctx-c->prefix)return 0;
    if(c->used+needed<=ctx)return 1;
    int drop=0,last=c->prefix;
    while(drop<c->turns&&c->used-(last-c->prefix)+needed>ctx)last=c->ends[drop++];
    if(c->used-(last-c->prefix)+needed>ctx)return 0;
    int removed=last-c->prefix;
    memmove(c->history+c->prefix,c->history+last,(size_t)(c->used-last)*sizeof(int));c->used-=removed;
    for(int i=drop;i<c->turns;i++)c->ends[i-drop]=c->ends[i]-removed;
    c->turns-=drop;
    int start=0;
    if(prefix_matches(c,c->history,c->prefix)){transfer_prefix(c,1);start=c->prefix;}
    else reset_state(c->s);
    prefill(c->s,c->history+start,c->used-start,0);
    return 1;
}
static int chat_turn(Chat *c,const char *text,int maximum,float temp,float topp,Candidate *candidates,int ipc){int ctx=c->s->m->ctx;char *enriched=memory_context(text);int n=role_message(c,"user",enriched,1,c->scratch,ctx);free(enriched);if(n<0)return -1;int k=role_message(c,"assistant","",0,c->scratch+n,ctx-n);if(k<0)return -1;n+=k;
    int room=ctx-c->prefix-n-3;if(room<1)return -1;if(maximum>room)maximum=room;if(!make_room(c,n+maximum+3))return -1;append(c,c->scratch,n,1);int count=0;
    for(int i=0;i<maximum;i++){if(stop_requested())break;int token=sample(c->s,temp,topp,candidates);if(token==c->s->m->eos)break;int len,special;const unsigned char *raw=slm_token_bytes(c->tok,token,&len,&special);if(special&&!c->s->m->arch)break;if(!special&&raw)fwrite(raw,1,(size_t)len,stdout);fflush(stdout);count++;append(c,&token,1,1);}
    int ending[16];ending[0]=c->s->m->eos;int en=1+encode_text(c->tok,"\n",ending+1,15);append(c,ending,en,0);if(c->turns>=128)die("too many history turn boundaries");c->ends[c->turns++]=c->used;if(ipc)printf("\n\001EOT %d\n",count);else printf("\n");fflush(stdout);return count;
}
static void usage(void){fprintf(stderr,"Usage: slm_run MODEL.SLM TOKENIZER.SLT [-c context] [-t temp] [-p top_p] [-n max_tokens]\n  [-s seed] [-m MEMORY.TXT] [-j threads] [--float-activations] [--prefix-cache path]\n  [--prompt text] [--raw text] [--tokens comma_ids] [--logits output.f32]\n  [--tokenize text] [--specials] [--system text]\n  Without a prompt, serves the Bliss Win32 sentinel protocol on stdin/stdout.\n");}
int main(int argc,char **argv){
    if(argc<3){usage();return 2;}const char *modelpath=argv[1],*tokpath=argv[2],*prompt=NULL,*raw=NULL,*tokenlist=NULL,*logitpath=NULL,*tokenize=NULL,*prefixpath=NULL,*sysprompt="You are Bliss, a helpful local assistant running on Windows XP. Give a clear, concise answer, usually in one to three sentences. Use the conversation and supplied notes when relevant. If the supplied information does not answer the question, say so.";int ctx=0,maximum=96,specials=0,threads=0;float temp=0.0f,topp=0.9f;
    for(int i=3;i<argc;i++){const char *a=argv[i];if(!strcmp(a,"--specials")){specials=1;continue;}if(!strcmp(a,"--float-activations")){float_activations=1;continue;}if(i+1>=argc){usage();return 2;}const char *v=argv[++i];if(!strcmp(a,"-c")||!strcmp(a,"--ctx"))ctx=atoi(v);else if(!strcmp(a,"-t")||!strcmp(a,"--temp"))temp=(float)atof(v);else if(!strcmp(a,"-p")||!strcmp(a,"--top-p"))topp=(float)atof(v);else if(!strcmp(a,"-n")||!strcmp(a,"--max-tokens"))maximum=atoi(v);else if(!strcmp(a,"--prompt"))prompt=v;else if(!strcmp(a,"--raw"))raw=v;else if(!strcmp(a,"--tokens"))tokenlist=v;else if(!strcmp(a,"--logits"))logitpath=v;else if(!strcmp(a,"--tokenize"))tokenize=v;else if(!strcmp(a,"--system"))sysprompt=v;else if(!strcmp(a,"--prefix-cache"))prefixpath=v;else if(!strcmp(a,"-j")||!strcmp(a,"--threads"))threads=atoi(v);else if(!strcmp(a,"-s")||!strcmp(a,"--seed"))rngstate=(uint32_t)strtoul(v,NULL,10);else if(!strcmp(a,"-m"))snprintf(memories.path,sizeof(memories.path),"%s",v);else{usage();return 2;}}
    if(maximum<1||maximum>8192||!isfinite(temp)||temp<0||!isfinite(topp)||topp<=0||topp>1)die("invalid sampling settings");
    if(!rngstate)rngstate=42;
    if(threads<0||threads>64)die("invalid thread count");
    slm_threads_init(threads);
    slm_tokenizer *tok=slm_tokenizer_load(tokpath);if(!tok)die("cannot load tokenizer");
    if(tokenize){int cap=(int)strlen(tokenize)+32,*ids=alloc((size_t)cap,sizeof(int));int n=slm_encode(tok,tokenize,specials,ids,cap);if(n<0)die("tokenizer failed");printf("[");for(int i=0;i<n;i++)printf("%s%d",i?",":"",ids[i]);printf("]\n");free(ids);slm_tokenizer_free(tok);return 0;}
    Model *m=load_model(modelpath,ctx);if(slm_tokenizer_vocab(tok)!=m->vocab)die("model/tokenizer vocab mismatch");State *s=new_state(m);Candidate *candidates=alloc((size_t)m->vocab,sizeof(Candidate));
    fprintf(stderr,"[slm] %d layers dim%d Q%d KV%d context%d weights%.2fMiB KV%.2fMiB backend=%s\n",m->nl,m->d,m->expanded?6:m->bits,m->nk,m->ctx,(double)m->mapsize/1048576.0,(double)(2ull*m->na*m->ctx*m->nk*m->hd*4)/1048576.0,SIMD?"SSE2":"scalar");
    if(raw||tokenlist){int *ids=alloc((size_t)m->ctx,sizeof(int)),n=0;if(raw)n=slm_encode(tok,raw,specials,ids,m->ctx);else{const char *p=tokenlist;while(*p&&n<m->ctx){char *end;long x=strtol(p,&end,10);if(end==p||x<0||x>=m->vocab)die("invalid token list");ids[n++]=(int)x;if(!*end)break;if(*end!=',')die("invalid token list separator");p=end+1;}}if(n<1)die("empty or oversized prompt");prefill(s,ids,n,1);if(logitpath){FILE *f=fopen(logitpath,"wb");if(!f||fwrite(s->logits,4,(size_t)m->vocab,f)!=(size_t)m->vocab)die("cannot write logits");fclose(f);}else{for(int j=0;j<maximum&&s->pos<m->ctx;j++){int token=sample(s,temp,topp,candidates);if(token==m->eos)break;int len,sp;const unsigned char *b=slm_token_bytes(tok,token,&len,&sp);if(!sp)fwrite(b,1,(size_t)len,stdout);forward(s,token,1);}printf("\n");}free(ids);}
    else{Chat c={0};c.s=s;c.tok=tok;c.history=alloc((size_t)m->ctx,sizeof(int));c.scratch=alloc((size_t)m->ctx,sizeof(int));prefix_disk_init(&c,prefixpath,argv[0]);snprintf(c.system,sizeof(c.system),"%s",sysprompt);if(!memory_load())fprintf(stderr,"[slm] cannot read persistent notes\n");if(!chat_reset(&c))die("system prompt exceeds half the context");
        if(prompt){if(chat_turn(&c,prompt,maximum,temp,topp,candidates,0)<0)die("prompt too long for context");}
        else{setvbuf(stdin,NULL,_IONBF,0);setvbuf(stdout,NULL,_IONBF,0);
#ifdef _WIN32
            _setmode(_fileno(stdin),_O_BINARY);_setmode(_fileno(stdout),_O_BINARY);
#endif
            printf("\001READY\n\001INFO MODEL %s native Q%d%s SSE2; context%d; %d thread%s\n\001INFO SOURCE %s\n",m->arch?"LFM2.5":"SmolLM2",m->expanded?6:m->bits,m->expanded?"X4":"",m->ctx,slm_pool.count,slm_pool.count==1?"":"s",m->arch?"LiquidAI/LFM2.5-350M":"HuggingFaceTB/SmolLM2-360M-Instruct");char line[32768];
            while(1){if(pending[0]){snprintf(line,sizeof(line),"%s",pending);pending[0]=0;}else if(!fgets(line,sizeof(line),stdin))break;line[strcspn(line,"\r\n")]=0;if(!line[0])continue;if(!strcmp(line,"\001STOP"))continue;
                if(line[0]=='/'){char *arg=strchr(line,' ');if(arg)*arg++=0;
                    if(!strcmp(line,"/reset"))chat_reset(&c);
                    else if(!strcmp(line,"/temp")&&arg){float v=(float)atof(arg);if(isfinite(v)&&v>=0&&v<=5)temp=v;}
                    else if(!strcmp(line,"/topp")&&arg){float v=(float)atof(arg);if(isfinite(v)&&v>0&&v<=1)topp=v;}
                    else if(!strcmp(line,"/maxtok")&&arg){int v=atoi(arg);if(v>0&&v<=8192)maximum=v;}
                    else if(!strcmp(line,"/seed")&&arg){rngstate=(uint32_t)strtoul(arg,NULL,10);if(!rngstate)rngstate=42;}
                    else if(!strcmp(line,"/system")&&arg){unescape(arg);int n=system_message(&c,arg,c.scratch,m->ctx);if(strlen(arg)>=sizeof(c.system)||n<0||n>m->ctx/2)printf("\001ERR System prompt exceeds half the context; previous prompt retained.\n");else{snprintf(c.system,sizeof(c.system),"%s",arg);chat_reset(&c);}}
                    else if(!strcmp(line,"/defaults")||!strcmp(line,"/preset")){temp=0.0f;topp=0.9f;maximum=96;}
                    else if(!strcmp(line,"/remember")&&arg){char note[MEM_LINE];clean_note(note,arg);int duplicate=0;for(int j=0;j<memories.count;j++)if(!strcmp(note,memories.notes[j]))duplicate=1;if(!note[0])printf("\001ERR Nothing to remember.\n");else if(duplicate)printf("\001INFO This note is already saved.\n");else if(memories.count>=MEM_MAX)printf("\001ERR Memory full; /forget a note first.\n");else{snprintf(memories.notes[memories.count++],MEM_LINE,"%s",note);if(!memory_save()){memories.count--;printf("\001ERR Could not save note; no change made.\n");}else printf("\001INFO Saved note %d: %s\n",memories.count,note);}}
                    else if(!strcmp(line,"/memories")){if(!memories.count)printf("\001INFO No notes saved.\n");for(int j=0;j<memories.count;j++)printf("\001INFO %d. %s\n",j+1,memories.notes[j]);}
                    else if(!strcmp(line,"/forget")&&arg){int index=atoi(arg)-1;if(index<0||index>=memories.count)printf("\001ERR No such note; use /memories.\n");else{Memories old=memories;for(int j=index;j+1<memories.count;j++)memcpy(memories.notes[j],memories.notes[j+1],MEM_LINE);memories.count--;if(!memory_save()){memories=old;printf("\001ERR Could not save notes; no change made.\n");}else printf("\001INFO Forgot note %d.\n",index+1);}}
                    else if(!strcmp(line,"/memreload")){if(memory_load())printf("\001INFO Loaded %d notes.\n",memories.count);else printf("\001ERR Could not read notes.\n");}
                    else if(!strcmp(line,"/memfile")&&arg){Memories old=memories;snprintf(memories.path,sizeof(memories.path),"%s",arg);if(!memory_load()){memories=old;printf("\001ERR Could not read notes file; previous notes retained.\n");}else printf("\001INFO Loaded %d notes.\n",memories.count);}
                    else if(!strcmp(line,"/replay")&&arg){char *sep=strchr(arg,'\t');if(sep){*sep++=0;unescape(arg);unescape(sep);int n=role_message(&c,"user",arg,1,c.scratch,m->ctx);if(n>0){int k=role_message(&c,"assistant",sep,1,c.scratch+n,m->ctx-n);if(k>0&&make_room(&c,n+k)){append(&c,c.scratch,n+k,0);c.ends[c.turns++]=c.used;}}}}
                    else if(!strcmp(line,"/info")||!strcmp(line,"/help")||!strcmp(line,"/template"))printf("\001INFO ChatML, Q%d%s, context%d, temp%.2f, top_p%.2f, maxtok%d; /reset /temp /topp /maxtok /seed /system /replay\n",m->expanded?6:m->bits,m->expanded?"X4":"",m->ctx,temp,topp,maximum);
                    else printf("\001ERR unsupported command: %s\n",line);
                    printf("\001EOT 0\n");continue;
                }
                unescape(line);if(chat_turn(&c,line,maximum,temp,topp,candidates,1)<0)printf("\001ERR Prompt too long for the configured context.\n\001EOT 0\n");
            }
        }free_prefix(&c);free(c.history);free(c.scratch);
    }
    free(candidates);free_state(s);free_model(m);slm_tokenizer_free(tok);return 0;
}
