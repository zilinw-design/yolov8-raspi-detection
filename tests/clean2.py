"""Remove orphaned CSS lines after _serve_html and before _serve_test_image."""
lines = open('src/phase3/screen_measure.py', 'r', encoding='utf-8').readlines()
out = []
skip = False
for i, line in enumerate(lines):
    # Check if this is the end of _serve_html (line after wfile.write(html.encode()))
    # and next line is orphaned CSS
    if i > 0 and 'self.wfile.write(html.encode())' in lines[i-1] and line.strip().startswith('"'):
        skip = True
        continue
    if skip:
        if line.strip().startswith('def '):
            skip = False
            out.append(line)
        continue
    out.append(line)

open('src/phase3/screen_measure.py', 'w', encoding='utf-8').writelines(out)
print('done')
