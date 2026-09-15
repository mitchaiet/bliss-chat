"""Exercise real bounded retrieval code, UTF-8 input validation and disk reads."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from test_xpchat_transport import excerpt

ROOT = Path(__file__).resolve().parents[1]

class DocumentRetrievalTests(unittest.TestCase):
    def test_actual_retrieval_helpers_with_sanitizers(self):
        source = (ROOT / 'src/xpchat.c').read_text()
        parts = [
            excerpt(source, 'static int utf8_unit(', 'static WCHAR *utf8_to_wide('),
            excerpt(source, 'static void trim_incomplete_utf8_tail(', 'static void replace_selection_utf8('),
            excerpt(source, '#define KNOW_TERMS_MAX', 'static void knowledge_scan_dir('),
        ]
        code = '#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <strings.h>\n#include <ctype.h>\n#include <assert.h>\n#define MAX_PATH 260\n#define _stricmp strcasecmp\n'
        code += '\n'.join(parts).replace('static char gKnowledgeSources[4 * MAX_PATH + 32];', '')
        code += r'''
int main(void) {
    char terms[KNOW_TERMS_MAX][KNOW_TERM_LEN], passage[KNOW_SNIPPET_MAX + 1];
    int n = know_extract_terms("Where is the ZEPHYR museum? Zephyr museum hours", terms, KNOW_TERMS_MAX);
    assert(n == 3);
    assert(know_find_term("concatenate", "cat") == -1);
    assert(know_find_term("catfish", "cat") == -1);
    assert(know_find_term("A cat.", "cat") == 2);
    char doc[4000] = "Zephyr once appeared here. ";
    for (int i=0;i<50;i++) strcat(doc, "Unrelated text padding without query words. ");
    strcat(doc, "The Zephyr museum hours are 10 AM to 4 PM, Tuesday through Friday.");
    assert(know_best_passage(doc, terms, n, passage) == 3);
    assert(strstr(passage, "10 AM to 4 PM"));
    assert(know_valid_text("caf\xc3\xa9 \xe2\x98\x83", 9));
    assert(!know_valid_text("\xc0\xaf", 2));
    assert(!know_valid_text("a\0b", 3));
    assert(!know_valid_text("\xed\xa0\x80", 3));
    assert(!know_valid_text("\xf0\x9f\x98", 3));
    assert(!know_valid_text("\xff\xfe", 2));
    char unicode[1024] = "";
    for(int i=0;i<90;i++) strcat(unicode, "caf\xc3\xa9 \xe2\x98\x83 ");
    for(int i=0;i<700;i++) {
        know_extract_snippet(unicode, i, passage, sizeof(passage));
        assert(know_valid_text(passage, strlen(passage)));
        assert(strlen(passage) <= KNOW_SNIPPET_MAX);
    }
    assert(know_extension("notes.MD") && know_extension("page.HTML"));
    assert(!know_extension("document.pdf") && !know_extension("file.txt.exe"));
    FILE *f=fopen("bom.txt", "wb"); assert(f);
    fputs("\xef\xbb\xbfPlain text",f); fclose(f);
    char *text=know_read_file("bom.txt"); assert(text && !strcmp(text,"Plain text")); free(text);
    f=fopen("page.html","wb"); fputs("<p>Opening <b>hours</b></p>",f); fclose(f);
    text=know_read_file("page.html"); assert(text && strstr(text,"Opening hours")); free(text);
    f=fopen("large.txt","wb"); for(int i=0;i<KNOW_FILE_MAX;i++) fputc('x',f); fclose(f);
    text=know_read_file("large.txt"); assert(text && strlen(text)==KNOW_FILE_MAX); free(text);
    f=fopen("large.txt","ab"); fputc('x',f); fclose(f);
    assert(!know_read_file("large.txt"));
    f=fopen("binary.txt","wb"); fwrite("a\0b",1,3,f); fclose(f);
    assert(!know_read_file("binary.txt"));
    assert(!know_read_file("missing.txt"));
    puts("DOCUMENT_RETRIEVAL_CHECKS passed (including 700 UTF-8 snippet boundaries)");
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='bliss-doc-test-') as d:
            p = Path(d); (p/'test.c').write_text(code)
            compile = subprocess.run([os.environ.get('XPCHAT_TEST_CC','clang'), '-std=c99', '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined', str(p/'test.c'), '-o', str(p/'test')], capture_output=True, text=True)
            self.assertEqual(compile.returncode, 0, compile.stderr)
            run = subprocess.run([str(p/'test')], cwd=d, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            print(run.stdout.strip())

if __name__ == '__main__':
    unittest.main()
