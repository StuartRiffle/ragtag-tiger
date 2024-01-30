# RAG/TAG Tiger - extensions.py
# Copyright (c) 2024 Stuart Riffle

built_in_loaders = set([
    # SimpleDirectoryReader supports these out-of-the-box
    '.pdf', '.csv', '.md', '.mbox', '.ipynb',
    '.docx', '.epub', '.hwp', '.ppt', '.pptm', '.pptx',
    '.jpeg', '.jpg', '.png', '.mp3', '.mp4', # ?
])

available_hub_loaders = {
    # Known custom loaders available on https://llamahub.ai
    ".json":    "JSONReader",
    ".xlsx":    "PandasExcelReader",
    ".graphql": "SDLReader",
    ".gql":     "SDLReader",
}

mime_file_types = set([
    # Pull out embedded text/html and uuencoded files
    ".eml", ".msg",

    # Special case: sometimes .doc files are actually MIME, not old binary Word documents
    ".doc",
])

archive_file_types = set([
    # Unpack these archive formats so we can index their contents too
    ".zip", ".7z", ".tar", ".gz", ".tgz", ".bz2", ".tbz2", ".xz", ".txz", 

    # FIXME - these common but unsupported here
    # ".rar", ".lzma", ".lz", ".lz4", ".zst", 
])

chunk_as_text = set([
    # Plain text files, no special handling 
    ".txt", ".TXT", ".rtf", ".log", ".asc", ".ini", ".cfg", 

    # FIXME - use a proper splitter/loader 
    ".hlsl", ".hlsli", ".fxh", ".glsl", ".glsli", ".shader",
    ".asm", ".s",
    ".xml",
])

from llama_index.text_splitter import CodeSplitter
source_code_splitters = [
    # Source code files get syntax-aware chunking
    ([".c", ".h"],          CodeSplitter(language="cpp")),            
    ([".cl"],               CodeSplitter(language="commonlisp")),
    ([".cpp", ".hpp"],      CodeSplitter(language="cpp")),
    ([".cxx", ".hxx"],      CodeSplitter(language="cpp")),
    ([".cs"],               CodeSplitter(language="c_sharp")),
    ([".css"],              CodeSplitter(language="css")),
    ([".dockerfile"],       CodeSplitter(language="dockerfile")),
    ([".dot"],              CodeSplitter(language="dot")),
    ([".el", ".emacs"],     CodeSplitter(language="elisp")),
    ([".elm"],              CodeSplitter(language="elm")),
    ([".ex", ".exs"],       CodeSplitter(language="elixir")),
    ([".f", ".f90"],        CodeSplitter(language="fortran")),
    ([".go"],               CodeSplitter(language="go")),
    ([".hs"],               CodeSplitter(language="haskell")),
    ([".html", ".htm"],     CodeSplitter(language="html")),
    ([".inc", ".inl"],      CodeSplitter(language="cpp")),
    ([".java"],             CodeSplitter(language="java")),
    ([".jl"],               CodeSplitter(language="julia")),
    ([".js"],               CodeSplitter(language="javascript")),
    ([".kt", ".kts"],       CodeSplitter(language="kotlin")),
    ([".lisp", ".lsp"],     CodeSplitter(language="commonlisp")),
    ([".lua"],              CodeSplitter(language="lua")),
    ([".m"],                CodeSplitter(language="objc")),
    ([".ml", ".mli"],       CodeSplitter(language="ocaml")),
    ([".php"],              CodeSplitter(language="php")),
    ([".pl"],               CodeSplitter(language="perl")),
    ([".py"],               CodeSplitter(language="python")),
    ([".r"],                CodeSplitter(language="r")),
    ([".rb"],               CodeSplitter(language="ruby")),
    ([".rs"],               CodeSplitter(language="rust")),
    ([".scala"],            CodeSplitter(language="scala")),
    ([".sh"],               CodeSplitter(language="bash")),
    ([".sql"],              CodeSplitter(language="sql")),
    ([".sqlite"],           CodeSplitter(language="sqlite")),
    ([".ts"],               CodeSplitter(language="typescript")),
    ([".yaml", ".yml"],     CodeSplitter(language="yaml")),
]

