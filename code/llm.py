# RAG/TAG Tiger - llm.py
# Copyright (c) 2024 Stuart Riffle
# github.com/stuartriffle/ragtag-tiger

import os
import torch
from files import *
from lograg import lograg, lograg_verbose, lograg_error
from timer import TimerUntil

openai_model_default    = "gpt-3.5-turbo-instruct"
google_model_default    = "models/text-bison-001"
anthropic_model_default = "claude-2"
perplexity_default      = "llama-2-70b-chat"
replicate_default       = "mistralai/mixtral-8x7b-instruct-v0.1"
default_timeout         = 180
hf_model_nicknames      = { "default": "codellama/CodeLlama-7b-Instruct-hf" }
default_llm_provider    = "huggingface"



def set_global_service_context_for_llm(llm, embed_model="local"):
    """Set the global service context to use a specific LLM and embed model"""
    from llama_index import ServiceContext, set_global_service_context
    service_context = ServiceContext.from_defaults(
        embed_model=embed_model, 
        llm=llm)
    set_global_service_context(service_context)
    return service_context

def load_llm(provider, model, server, api_key, params, verbose=False, set_service_context=True):
    result = None
    streaming_supported = True
    try:
        with TimerUntil("ready"):
            model_kwargs = dict([param.split("=") for param in params]) if params else {}

            ### OpenAI
            if provider == "openai":
                api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
                if not server:
                    model_name = model or openai_model_default
                    lograg(f"Preparing OpenAI model \"{model_name}\"...")
                    from llama_index.llms import OpenAI
                    result = OpenAI(
                        model=model_name,
                        timeout=default_timeout,
                        api_key=api_key,
                        additional_kwargs=model_kwargs,
                        verbose=verbose)
                else:
                    # API compatible server
                    model_name = model or "default"
                    lograg(f"Preparing model \"{model_name}\" on server \"{server}\"...")
                    from llama_index.llms import OpenAILike
                    result = OpenAILike(
                        model=model_name,
                        additional_kwargs=model_kwargs,
                        api_base=server,
                        max_tokens=1000,
                        max_iterations=100,
                        timeout=default_timeout,
                        verbose=verbose)
                
            ### Google
            elif provider == "google":
                api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
                model_name = model or google_model_default
                lograg(f"Preparing Google model \"{model_name}\"...")
                from llama_index.llms import PaLM
                result = PaLM(
                    api_key=api_key,
                    model_name=model_name,
                    generate_kwargs=model_kwargs)
                streaming_supported = False
                
            ### Llama.cpp
            elif provider == "llamacpp":
                if torch.cuda.is_available():
                    # FIXME - this does nothing?
                    model_kwargs["n_gpu_layers"] = -1
                lograg(f"Preparing llama.cpp model \"{os.path.normpath(model)}\"...")
                from llama_index.llms import LlamaCPP
                result = LlamaCPP(
                    model_path=model,
                    model_kwargs=model_kwargs,
                    verbose=verbose)
                
            ### Perplexity
            elif provider == "perplexity":
                api_key = api_key or os.environ.get("PERPLEXITYAI_API_KEY", "")
                model_name = model or perplexity_default
                lograg(f"Preparing Perplexity model \"{model_name}\"...")
                from llama_index.llms import Perplexity
                result = Perplexity(
                    api_key=api_key,
                    model=model_name,
                    model_kwargs=model_kwargs)
                
            ### Replicate
            elif provider == "replicate":
                api_key = api_key or os.environ.get("REPLICATE_API_TOKEN", "")
                model_name = model or replicate_default
                lograg(f"Preparing Replicate model \"model_name)\"...")
                from llama_index.llms import Replicate
                result = Replicate(
                    model=model_name,
                    additional_kwargs=model_kwargs)
            
            ### HuggingFace
            else:
                os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
                model_desc = ""
                model_name = model or "default"
                if model_name in hf_model_nicknames:
                    model_desc = f" (\"{model_name}\")"
                    model_name = hf_model_nicknames[model_name]
                lograg(f"Preparing HuggingFace model \"{model_name}\"{model_desc}...")

                from llama_index.llms import HuggingFaceLLM
                result = HuggingFaceLLM(
                    model_name=model_name,
                    model_kwargs=model_kwargs) 
                    #device_map=args.torch_device or "auto",
                    #system_prompt=system_prompt)

            if set_service_context:
                set_global_service_context_for_llm(result)

    except Exception as e: 
        lograg_error(f"failure initializing LLM: {e}", exit_code=1)

    return result, streaming_supported



def split_llm_config(config):
    """Split an LLM from a config string like "provider,model,server,api-key,param1,param2,..." into its components"""
    config   = config.strip("\"' ")
    fields   = config.split(",")
    provider = fields[0].strip() if len(fields) > 0 else default_llm_provider
    model    = fields[1].strip() if len(fields) > 1 else None
    server   = fields[2].strip() if len(fields) > 2 else None
    api_key  = fields[3].strip() if len(fields) > 3 else None
    params   = fields[4:]        if len(fields) > 4 else []
    return provider, model, server, api_key, params


def load_llm_config(config, set_service_context=True):
    """Load an LLM from a config string like "provider,model,server,api-key,param1,param2,..."""
    provider, model, server, api_key, params = split_llm_config(config)
    return load_llm(provider.lower(), model, server, api_key, params, set_service_context)


