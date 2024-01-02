class CodeAwareDirectoryReader(SimpleDirectoryReader):
    """Chunks source code files in a roughly language-aware way."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.custom_splitters = [
            ([".cpp", ".c", ".hpp", ".h"],  CodeSplitter("cpp")),
            ([".cs"],                       CodeSplitter("c-sharp")),
            ([".py"],                       CodeSplitter("python")),
            ([".lua"],                      CodeSplitter("lua")),
            ([".cu"],                       CodeSplitter("cuda")),
            ([".java"],                     CodeSplitter("java")),
            ([".js"],                       CodeSplitter("javascript")),
            ([".ts"],                       CodeSplitter("typescript")),
        ]        

    def readFile(self, file_path):
        for extensions, code_splitter in self.custom_splitters:
            if file_path.endswith(tuple(extensions)):
                with open(file_path, 'r') as f:
                    source_code = f.read()
                    chunks = code_splitter.chunk(source_code)
                    docs = [Document(file_path, chunk) for chunk in chunks]
                    return docs
                
        return super().readFile(file_path)
    