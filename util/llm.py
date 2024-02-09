# RAG/TAG Tiger - llm.py
# Copyright (c) 2024 Stuart Riffle
# github.com/stuartriffle/ragtag-tiger

import os
import torch
from .files import *
from .lograg import lograg, lograg_verbose, lograg_error
from .timer import TimerUntil

openai_model_default    = "gpt-3.5-turbo-instruct"
google_model_default    = "models/text-bison-001"
anthropic_model_default = "claude-2"
mistral_default         = "mistral-small"
perplexity_default      = "llama-2-70b-chat"
replicate_default       = "mistralai/mixtral-8x7b-instruct-v0.1"
fireworks_ai_default    = "accounts/fireworks/models/mixtral-8x7b-instruct"
together_ai_default     = "codellama/CodeLlama-70b-Instruct-hf"

default_timeout         = 180
default_temperature     = 0.1
default_max_tokens      = 500
default_llm_provider    = "huggingface"
hf_model_nicknames      = { "default": "codellama/CodeLlama-7b-Instruct-hf" }

def load_llm(provider, model, server, api_key, params, global_params, verbose=False, set_service_context=True, torch_device=None):
    result = None
    streaming_supported = True
    try:
        with TimerUntil("ready"):
            all_params = global_params.copy()
            model_params = dict([param.split("=") for param in params]) if params else {}
            for k, v in model_params.items():
                all_params[k] = v

            model_kwargs = {}
            for k, v in all_params.items():
                model_kwargs[k] = float(v) if v.replace(".", "", 1).isdigit() else v

            temperature = float(model_kwargs.get("temperature", default_temperature))
            max_tokens = int(model_kwargs.get("max_tokens", default_max_tokens))

            ### OpenAI
            if provider == "openai" and not server:
                model_name = model or openai_model_default
                api_key = api_key or os.environ.get("OPENAI_API_KEY", "") 
                lograg(f"OpenAI model \"{model_name}\"...")
                from llama_index.llms import OpenAI
                result = OpenAI(
                    model=model_name,
                    timeout=default_timeout,
                    api_key=api_key,
                    additional_kwargs=model_kwargs,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    verbose=verbose)
                
            ### OpenAI API-compatible third party server                
            elif provider == "openai" and server:
                # Auto-populate API key and model for known providers
                if "together.ai" in server or "together.xyz" in server:
                    api_key = api_key or os.environ.get("TOGETHERAI_API_KEY", "")
                    model = model or together_ai_default
                if "fireworks.ai" in server: 
                    api_key = api_key or os.environ.get("FIREWORKS_API_KEY", "")
                    model = model or fireworks_ai_default
                api_key = api_key or os.environ.get("OPENAI_API_KEY", "") 
                model_name = model or "default"
               
                lograg(f"Model \"{model_name}\" on \"{server}\"...")
                from llama_index.llms import OpenAILike
                result = OpenAILike(
                    api_key=api_key,
                    model=model_name,
                    additional_kwargs=model_kwargs,
                    api_base=server,
                    max_iterations=100,
                    timeout=default_timeout,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    verbose=verbose)
                
            ### Google
            elif provider == "google":
                gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
                google_api_key = os.environ.get("GOOGLE_API_KEY", "")
                model_name = model or google_model_default
                import google.generativeai as genai
                genai.configure(api_key=google_api_key)

                if "gemini" in str(model_name).lower():
                    lograg(f"Google Gemini model \"{model_name}\"...")
                    from llama_index.llms import Gemini
                    result = Gemini(
                        api_key=api_key or gemini_api_key,
                        model_name=model_name,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        model_kwargs=model_kwargs)
                else:
                    lograg(f"Google PaLM model \"{model_name}\"...")
                    from llama_index.llms import PaLM
                    result = PaLM(
                        api_key=api_key or google_api_key,
                        model_name=model_name,
                        generate_kwargs=model_kwargs)
                    streaming_supported = False
                
            ### Llama.cpp
            elif provider == "llamacpp":
                if torch.cuda.is_available():
                    # FIXME - this does nothing? Always on CPU
                    model_kwargs["n_gpu_layers"] = -1
                lograg(f"llama.cpp model \"{cleanpath(model)}\"...")
                from llama_index.llms import LlamaCPP
                result = LlamaCPP(
                    model_path=model,
                    model_kwargs=model_kwargs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    verbose=verbose)
                
            ### Mistral
            elif provider == "mistral":
                api_key = api_key or os.environ.get("MISTRAL_API_KEY", None)
                model_name = model or mistral_default
                lograg(f"Mistral model \"{model_name}\"...")
                from llama_index.llms import MistralAI
                result = MistralAI(
                    api_key=api_key,
                    model=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    additional_kwargs=model_kwargs)
                
            ### Perplexity
            elif provider == "perplexity":
                api_key = api_key or os.environ.get("PERPLEXITYAI_API_KEY", "")
                model_name = model or perplexity_default
                lograg(f"Perplexity model \"{model_name}\"...")
                from llama_index.llms import Perplexity
                result = Perplexity(
                    api_key=api_key,
                    model=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    model_kwargs=model_kwargs)
                
            ### Replicate
            elif provider == "replicate":
                api_key = api_key or os.environ.get("REPLICATE_API_TOKEN", "")
                model_name = model or replicate_default
                lograg(f"Replicate model \"{model_name}\"...")
                from llama_index.llms import Replicate
                result = Replicate(
                    model=model_name,
                    temperature=temperature,
                    additional_kwargs=model_kwargs)
            
            ### HuggingFace
            else:
                os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
                model_desc = ""
                model_name = model or "default"
                if model_name in hf_model_nicknames:
                    model_desc = f" (\"{model_name}\")"
                    model_name = hf_model_nicknames[model_name]
                lograg(f"HuggingFace model \"{model_name}\"{model_desc}...")

                from llama_index.llms import HuggingFaceLLM
                result = HuggingFaceLLM(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    max_new_tokens=max_tokens,
                    device_map=torch_device or "auto")
                    #system_prompt=system_prompt)

            from llama_index import ServiceContext, set_global_service_context
            service_context = ServiceContext.from_defaults(
                embed_model='local',
                llm=result)

            if set_service_context:
                set_global_service_context(service_context)

    except Exception as e: 
        lograg_error(f"failure initializing LLM: {e}", exit_code=1)

    return result, streaming_supported, service_context



def split_llm_config(config):
    """Split an LLM from a config string of format "[alias=]provider[,model[,server[,api-key[,parameters...]]]]" into its components"""

    fields   = config.strip("\"' ").split(",")
    provider = fields[0].strip() if len(fields) > 0 else default_llm_provider
    model    = fields[1].strip() if len(fields) > 1 else None
    server   = fields[2].strip() if len(fields) > 2 else None
    api_key  = fields[3].strip() if len(fields) > 3 else None
    params   = fields[4:]        if len(fields) > 4 else []

    alias = None
    if "=" in provider:
        alias, provider = provider.split("=", 1)
        provider = provider.strip()

    return provider, model, server, api_key, params, alias


def load_llm_config(config, global_params, set_service_context=True):
    """Load an LLM from a config string like "provider,model,server,api-key,param1,param2,..."""
    provider, model, server, api_key, params, _ = split_llm_config(config)
    return load_llm(provider.lower(), model, server, api_key, params, global_params, set_service_context)


