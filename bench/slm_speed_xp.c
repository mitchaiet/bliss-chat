/* XP decode benchmark; link -lpsapi. Touch the entire 512-token KV cache.
 * Report independent timers because a busy emulated XP guest can lose ticks. */
#define main slm_backend_main
#include "../src/slm_run.c"
#undef main
#include <psapi.h>
static double elapsed(LARGE_INTEGER a,LARGE_INTEGER b,LARGE_INTEGER hz){return (double)(b.QuadPart-a.QuadPart)/(double)hz.QuadPart;}
int main(int argc,char **argv){
    if(argc!=2)return 2;
    LARGE_INTEGER hz,a,b; if(!QueryPerformanceFrequency(&hz))return 3;
    Model *m=load_model(argv[1],512);State *s=new_state(m);
    size_t bytes=(size_t)m->na*m->ctx*m->nk*m->hd*4;
    memset(s->kc,1,bytes);memset(s->vc,1,bytes);
    FILE *report=fopen("C:\\BLISS-RELEASE-BENCH.TXT","w");
    if(!report)return 4;
    puts("Bliss Q6X4: full context memory allocated. Timing two passes.");fflush(stdout);
    for(int pass=0;pass<2;pass++){
        int pre=pass?64:3,decode=pass?32:8;
        reset_state(s);QueryPerformanceCounter(&a);DWORD tick=GetTickCount();
        for(int i=0;i<pre;i++)forward(s,(i*317+19)%m->vocab,0);
        QueryPerformanceCounter(&b);double pre_s=elapsed(a,b,hz);DWORD pre_tick=GetTickCount()-tick;
        QueryPerformanceCounter(&a);tick=GetTickCount();
        for(int i=0;i<decode;i++)forward(s,(i*773+31)%m->vocab,1);
        QueryPerformanceCounter(&b);double sec=elapsed(a,b,hz);DWORD decode_tick=GetTickCount()-tick;
        PROCESS_MEMORY_COUNTERS pm;MEMORYSTATUS ms;
        memset(&pm,0,sizeof(pm));pm.cb=sizeof(pm);memset(&ms,0,sizeof(ms));ms.dwLength=sizeof(ms);
        GetProcessMemoryInfo(GetCurrentProcess(),&pm,sizeof(pm));GlobalMemoryStatus(&ms);
        char line[1024];snprintf(line,sizeof(line),
          "Pass %d: %d tokens / %.3f QPC sec = %.6f tok/s\n  Tick clock %.3f sec (%.6f tok/s)\n  Prefill %d: %.3f QPC sec; %.3f tick sec\n  Working %.2f MiB; peak %.2f; private %.2f; free %.2f\n",
          pass,decode,sec,decode/sec,decode_tick/1000.0,decode/(decode_tick/1000.0),pre,pre_s,pre_tick/1000.0,
          pm.WorkingSetSize/1048576.0,pm.PeakWorkingSetSize/1048576.0,pm.PagefileUsage/1048576.0,ms.dwAvailPhys/1048576.0);
        fputs(line,stdout);fflush(stdout);fputs(line,report);fflush(report);
    }
#ifdef SLM_PROFILE
    {
        double total=0;for(int i=0;i<5;i++)total+=slm_prof[i];
        char line[512];snprintf(line,sizeof(line),
          "Profile (%d threads): linear %.1f%%  attention %.1f%%  swiglu %.1f%%  conv %.1f%%  other %.1f%%\n",
          slm_pool.count,100*slm_prof[0]/total,100*slm_prof[1]/total,100*slm_prof[2]/total,100*slm_prof[3]/total,100*slm_prof[4]/total);
        fputs(line,stdout);fputs(line,report);
    }
#endif
    fclose(report);free_state(s);free_model(m);return 0;
}
