"""Criterion 1 - LLM integration: local (Ollama) or API-based (Anthropic / OpenAI)."""

PROVIDERS = ["Ollama (local)", "Anthropic (API)", "OpenAI (API)"]
DEFAULT_MODELS = {
    "Ollama (local)": "llama3.2",
    "Anthropic (API)": "claude-sonnet-5-5",
    "OpenAI (API)": "gpt-4o-mini",
}


def get_llm(provider: str, model: str | None = None, temperature: float = 0.2):
    model = model or DEFAULT_MODELS[provider]
    if provider == "Ollama (local)":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=temperature)
    if provider == "Anthropic (API)":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature, max_tokens=1500)
    if provider == "OpenAI (API)":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)
    raise ValueError(f"Unknown provider: {provider}")
