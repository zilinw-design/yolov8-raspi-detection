"""Remove orphaned inline HTML from screen_measure.py after _serve_html edit."""
lines = open('src/phase3/screen_measure.py', 'r', encoding='utf-8').readlines()
out = []
skip = False
for i, line in enumerate(lines):
    if 'html = "<html><body><h1>dashboard.html not found</h1></body></html>"' in line:
        out.append(line)
        out.append('        self.send_response(200)\n')
        out.append('        self.send_header("Content-Type", "text/html; charset=utf-8")\n')
        out.append('        self.send_header("Content-Length", str(len(html.encode())))\n')
        out.append('        self.end_headers()\n')
        out.append('        self.wfile.write(html.encode())\n')
        skip = True
        continue
    if skip:
        if line.strip().startswith('"') and 'html' not in out[-1]:
            continue
        if line.strip().startswith('self.send_response') or line.strip().startswith('self.send_header') or line.strip().startswith('self.end_headers') or line.strip().startswith('self.wfile.write'):
            continue
        skip = False
    out.append(line)

open('src/phase3/screen_measure.py', 'w', encoding='utf-8').writelines(out)
print('cleaned')
