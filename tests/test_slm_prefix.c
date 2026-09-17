/* Exact, bounded system-prefix snapshots for reset and context compaction.
 * Build with src/slm_tokenizer.c and pass MODEL.SLM TOKENIZER.SLT. */
#define main slm_backend_main
#include "../src/slm_run.c"
#undef main
#include <assert.h>

static void replay(State *s,const Chat *c){
    reset_state(s);
    for(int i=0;i<c->used;i++)forward(s,c->history[i],0);
}
static void same_cache(const State *a,const State *b){
    const Model *m=a->m;size_t kd=(size_t)m->nk*m->hd;
    assert(a->pos==b->pos);
    for(int i=0;i<m->na;i++){
        size_t off=(size_t)i*m->ctx*kd,bytes=(size_t)a->pos*kd*sizeof(float);
        assert(!memcmp(a->kc+off,b->kc+off,bytes));
        assert(!memcmp(a->vc+off,b->vc+off,bytes));
    }
    size_t conv=(size_t)m->nc*m->d*m->conv_width*sizeof(float);
    if(conv)assert(!memcmp(a->conv_state,b->conv_state,conv));
}
static void same_continuation(Chat *c,State *reference){
    same_cache(c->s,reference);
    int token=19;
    for(int i=0;i<3;i++){
        forward(c->s,token,1);forward(reference,token,1);
        assert(!memcmp(c->s->logits,reference->logits,(size_t)c->s->m->vocab*sizeof(float)));
        token=sample(c->s,0,0.9f,NULL);
        assert(token==sample(reference,0,0.9f,NULL));
    }
}
static void add_turn(Chat *c,int n,int salt){
    for(int i=0;i<n;i++)c->scratch[i]=(i*317+salt)%c->s->m->vocab;
    append(c,c->scratch,n,0);c->ends[c->turns++]=c->used;
}
int main(int argc,char **argv){
    if(argc!=3)return 2;
    Model *m=load_model(argv[1],128);slm_tokenizer *tok=slm_tokenizer_load(argv[2]);assert(tok);
    Chat c={0};c.s=new_state(m);c.tok=tok;
    c.history=alloc((size_t)m->ctx,sizeof(int));c.scratch=alloc((size_t)m->ctx,sizeof(int));
    State *reference=new_state(m);
    const char *initial="You are Bliss, a helpful local assistant running on Windows XP. Give a clear, concise answer, usually in one to three sentences. Use the conversation and supplied notes when relevant. If the supplied information does not answer the question, say so.";
    snprintf(c.system,sizeof(c.system),"%s",initial);
    clock_t begin=clock();assert(chat_reset(&c));double cold=(double)(clock()-begin)/CLOCKS_PER_SEC;
    int initial_tokens=c.prefix;
    size_t extra=(2ull*m->na*c.prefix*m->nk*m->hd+(size_t)m->nc*m->d*m->conv_width)*sizeof(float)+(size_t)c.prefix*sizeof(int);
    replay(reference,&c);same_cache(c.s,reference);
    add_turn(&c,8,17);
    /* A cache hit must not execute any forward call. forward always overwrites x. */
    c.s->x[0]=NAN;begin=clock();assert(chat_reset(&c));double hit=(double)(clock()-begin)/CLOCKS_PER_SEC;
    assert(isnan(c.s->x[0]));assert(c.turns==0&&c.used==c.prefix);
    same_continuation(&c,reference);
    assert(chat_reset(&c));
    /* A changed prefix with the same token count must invalidate the snapshot. */
    snprintf(c.system,sizeof(c.system),"Answer A.");assert(chat_reset(&c));
    int previous=c.prefix;
    snprintf(c.system,sizeof(c.system),"Answer B.");c.s->x[0]=NAN;assert(chat_reset(&c));
    assert(c.prefix==previous&&!isnan(c.s->x[0]));
    assert(prefix_matches(&c,c.history,c.used));replay(reference,&c);same_continuation(&c,reference);
    assert(chat_reset(&c));
    add_turn(&c,24,31);add_turn(&c,24,37);add_turn(&c,24,41);
    int before=c.used;
    assert(make_room(&c,m->ctx-before+1));assert(c.turns==2&&c.used==before-24);
    replay(reference,&c);same_continuation(&c,reference);
    /* Reset after generation, then evict all turns: both restore without forward. */
    assert(chat_reset(&c));add_turn(&c,24,43);c.s->x[0]=NAN;
    assert(make_room(&c,m->ctx-c.prefix));assert(isnan(c.s->x[0])&&c.used==c.prefix&&c.turns==0);
    replay(reference,&c);same_continuation(&c,reference);
    if(m->arch){
        c.system[0]=0;assert(chat_reset(&c));assert(c.prefix==1&&c.history[0]==m->bos);
        add_turn(&c,8,47);c.s->x[0]=NAN;assert(chat_reset(&c));assert(isnan(c.s->x[0]));
        replay(reference,&c);same_continuation(&c,reference);
    }
    /* Too-large system prompts must leave the active cached prefix intact. */
    assert(chat_reset(&c));int old_used=c.used;float *old_cache=c.prefix_state;
    for(int i=0;i<400;i++)memcpy(c.system+i*5,"word ",5);c.system[2000]=0;
    assert(!chat_reset(&c));assert(c.used==old_used&&c.prefix_state==old_cache);
    printf("{\"bit_exact_logit_continuations\":%d,\"default_prefix_tokens\":%d,\"cache_extra_bytes\":%zu,\"initial_prefix_cpu_seconds\":%.6f,\"cached_reset_cpu_seconds\":%.6f,\"reset_skipped_forward\":true,\"changed_same_length_prefix_recomputed\":true,\"rollover_replayed_only_remaining_turns\":true,\"empty_prefix_verified\":%s,\"oversized_prompt_retained_cache\":true}\n",m->arch?5:4,initial_tokens,extra,cold,hit,m->arch?"true":"false");
    free_prefix(&c);free(c.history);free(c.scratch);free_state(c.s);free_state(reference);free_model(m);slm_tokenizer_free(tok);return 0;
}
