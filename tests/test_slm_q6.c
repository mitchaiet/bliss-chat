/* Independent signed-six-bit packing oracle, including every value in every lane. */
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../src/slm_float_dot.h"
int main(void){
    uint32_t seed=81721;float maximum=0;
    for(int trial=0;trial<8192;trial++){
        unsigned char storage[49]={0},*packed=storage+(trial%2);
        float x[64];double expected=0,sumabs=0;int values[64];
        for(int j=0;j<64;j++){
            seed=seed*1664525u+1013904223u;int value=(int)(seed%64)-32;
            if(trial<4096)value=j==trial/64?trial%64-32:0;
            seed=seed*1664525u+1013904223u;x[j]=((int)(seed%200001)-100000)/137.0f;
            values[j]=value;unsigned u=(unsigned)(value+32);
            packed[j/2]|=(unsigned char)((u&15)<<(4*(j%2)));
            packed[32+j/4]|=(unsigned char)((u>>4)<<(2*(j%4)));
            expected+=(double)value*x[j];sumabs+=fabs((double)value*x[j]);
        }
        for(int j=0;j<64;j++)if(slm_q6_value(packed,j)!=values[j]){fprintf(stderr,"Q6 decode mismatch\n");return 1;}
        float got=slm_dot_float(packed,x,6,64);double error=fabs((double)got-expected);
        if(error>maximum)maximum=(float)error;
        if(error>32*FLT_EPSILON*sumabs+0.00001){fprintf(stderr,"Q6 dot mismatch case%d: %.9g vs%.9g\n",trial,(double)got,expected);return 1;}
    }
    printf("8192 Q6 packing/FP32 dot oracle cases passed (%s), maxabs%.9g.\n",SLM_FLOAT_DOT_SIMD?"SSE2":"scalar",(double)maximum);
    return 0;
}
