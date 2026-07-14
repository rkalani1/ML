import os
import re

fixes = 0
for root, dirs, files in os.walk('docs'):
    for f in files:
        if f.endswith('.md'):
            filepath = os.path.join(root, f)
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
            except UnicodeDecodeError:
                with open(filepath, 'r', encoding='windows-1252') as file:
                    content = file.read()
                
            old_content = content
            content = content.replace('https://rkalani1.github.io/ML/', '/')
            content = re.sub(r'https?://rkalani1\.github\.io/ML/?', '/', content)
            
            # fix garbled text
            content = content.replace('â€¦', '...')
            content = content.replace('â€™', chr(39))
            content = content.replace('â€œ', chr(34))
            content = content.replace('â€', chr(34))
            content = content.replace('â€”', '-')
            content = content.replace('â€“', '-')
            
            if content != old_content:
                with open(filepath, 'w', encoding='utf-8') as file:
                    file.write(content)
                print('Fixed ' + filepath)
                fixes += 1
print('Fixed ' + str(fixes) + ' markdown files.')

with open('mkdocs.yml', 'r', encoding='utf-8', errors='replace') as file:
    mkdocs = file.read()
old_mkdocs = mkdocs
mkdocs = mkdocs.replace('â€¦', '...')
mkdocs = mkdocs.replace('…', '...')
if mkdocs != old_mkdocs:
    with open('mkdocs.yml', 'w', encoding='utf-8') as file:
        file.write(mkdocs)
    print('Fixed mkdocs.yml')
