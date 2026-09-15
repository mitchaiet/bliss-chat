#define WIN32_LEAN_AND_MEAN
#define _WIN32_WINNT 0x0501
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static BOOL CALLBACK child(HWND w, LPARAM parent) {
    RECT r,p; char cls[128]; WCHAR title[1000]; char utf8[4000];
    if(!IsWindowVisible(w))return 1;
    GetClassNameA(w,cls,sizeof(cls));GetWindowRect(w,&r);GetClientRect((HWND)parent,&p);
    MapWindowPoints(HWND_DESKTOP,(HWND)parent,(POINT*)&r,2);
    SendMessageW(w,WM_GETTEXT,1000,(LPARAM)title);
    WideCharToMultiByte(CP_UTF8,0,title,-1,utf8,sizeof(utf8),NULL,NULL);
    for(char *t=utf8;*t;t++)if(*t=='\r'||*t=='\n')*t=' ';
    printf("id=%d class=%s rect=%ld,%ld,%ld,%ld enabled=%d text=%s\n",GetDlgCtrlID(w),cls,r.left,r.top,r.right,r.bottom,IsWindowEnabled(w),utf8);
    return 1;
}
int main(int argc,char **argv) {
    HWND w=FindWindowA(NULL,argc>1?argv[1]:"Bliss Chat - Local Assistant");
    if(!w){fprintf(stderr,"window missing\n");return 2;}
    const char *mode=argc>2?argv[2]:"inspect";
    if(!strcmp(mode,"inspect"))EnumChildWindows(w,child,(LPARAM)w);
    else if(!strcmp(mode,"command"))PostMessageA(w,WM_COMMAND,atoi(argv[3]),0);
    else if(!strcmp(mode,"set")) {WCHAR text[8192];MultiByteToWideChar(CP_UTF8,0,argv[4],-1,text,8192);SendDlgItemMessageW(w,atoi(argv[3]),WM_SETTEXT,0,(LPARAM)text);}
    else if(!strcmp(mode,"select"))SendDlgItemMessageA(w,atoi(argv[3]),LB_SETCURSEL,atoi(argv[4]),0);
    else if(!strcmp(mode,"close"))PostMessageA(w,WM_CLOSE,0,0);
    else if(!strcmp(mode,"text")){WCHAR buf[64000]; char text[256000];SendDlgItemMessageW(w,atoi(argv[3]),WM_GETTEXT,64000,(LPARAM)buf);WideCharToMultiByte(CP_UTF8,0,buf,-1,text,sizeof(text),NULL,NULL);printf("%s\n",text);}
    else if(!strcmp(mode,"origin")){POINT p={0,0};ClientToScreen(w,&p);printf("%ld %ld\n",p.x,p.y);}
    else if(!strcmp(mode,"resize"))SetWindowPos(w,0,0,0,atoi(argv[3]),atoi(argv[4]),SWP_NOZORDER);
    return 0;
}
