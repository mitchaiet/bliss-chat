/* Compare every signed SIMD lane with independent scalar integer arithmetic. */
#define main slm_backend_main
#include "../src/slm_run.c"
#undef main
int main(void) {
    uint32_t seed=183461;
    for(int trial=0;trial<4096;trial++) {
        int8_t activations[64],q8[64];unsigned char q4[32];int expected4=0,expected8=0;
        memset(q4,0,sizeof(q4));
        for(int j=0;j<64;j++) {
            seed=seed*1664525u+1013904223u;int a=(int)(seed%255u)-127;
            seed=seed*1664525u+1013904223u;int b=(int)(seed%255u)-127;
            seed=seed*1664525u+1013904223u;int c=(int)(seed%16u)-8;
            if(trial==0){a=-127;b=-127;c=-8;}if(trial==1){a=127;b=127;c=7;}
            activations[j]=(int8_t)a;q8[j]=(int8_t)b;q4[j/2]|=(unsigned char)((c+8)<<(4*(j%2)));
            expected4+=a*c;expected8+=a*b;
        }
        for(int n=16;n<=64;n*=2){
            int e4=0,e8=0;
            for(int j=0;j<n;j++){e4+=((((q4[j/2]>>(4*(j%2)))&15)-8)*activations[j]);e8+=q8[j]*activations[j];}
            if(dot_quant(q4,activations,4,n)!=e4||dot_quant((const unsigned char*)q8,activations,8,n)!=e8){fprintf(stderr,"quantized dot mismatch at case%d group%d\n",trial,n);return 1;}
        }
        if(dot_quant(q4,activations,4,64)!=expected4||dot_quant((const unsigned char*)q8,activations,8,64)!=expected8)return 1;
    }
    printf("12288 Q4 and12288 Q8 independent integer dot checks passed (%s).\n",SIMD?"SSE2":"scalar");return 0;
}
