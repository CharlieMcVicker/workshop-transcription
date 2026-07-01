import re

with open("frontend/src/App.jsx", "r", encoding="utf-8") as f:
    content = f.read()

# Replace hardcoded modern Tailwind classes
content = re.sub(r'\banimate-fade-in\b', '', content)
content = re.sub(r'\brounded-2xl\b', '', content)
content = re.sub(r'\brounded-xl\b', '', content)
content = re.sub(r'\brounded-lg\b', '', content)
content = re.sub(r'\brounded\b', '', content)
content = re.sub(r'\bshadow-lg\b', '', content)
content = re.sub(r'\bshadow-sm\b', '', content)
content = re.sub(r'\bshadow\b', '', content)
content = re.sub(r'\bfocus:ring-4\b', '', content)
content = re.sub(r'\btransition-all\b', '', content)
content = re.sub(r'\btransform\b', '', content)
content = re.sub(r'\bactive:scale-\[0\.99\]\b', '', content)
content = re.sub(r'\btransition-colors\b', '', content)
content = re.sub(r'\bduration-200\b', '', content)
content = re.sub(r'\bhover:scale-110\b', '', content)
content = re.sub(r'\bactive:scale-95\b', '', content)
content = re.sub(r'\bmax-w-3xl\b', 'max-w-3xl mx-auto text-center', content)
content = re.sub(r'  +', ' ', content) # compact extra spaces optionally
with open("frontend/src/App.jsx", "w", encoding="utf-8") as f:
    f.write(content)
